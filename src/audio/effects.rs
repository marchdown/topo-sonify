/// Upper bound on the summed feedback gain of the delay network (see `set_from_persistence`).
const MAX_LOOP_GAIN: f64 = 0.5;

/// Upper bound on the summed level of the wet taps, so the echo bus is at most half as loud
/// as the loop it reads from.
///
/// Together the two bounds give |out| <= a * (1 + MAX_WET_LEVEL / (1 - MAX_LOOP_GAIN)) = 2a
/// for any input bounded by a, so the clamp in `process` stays a guard for a loud input and
/// never shapes the sound of a quiet one.
const MAX_WET_LEVEL: f64 = 0.5;

/// Cap on the number of wavefold stages in `BettiDistortion`. Each stage has small-signal gain
/// pi/2, so the stage count sets an exponential gain, (pi/2)^stages. Two stages keep it under 2.5.
const MAX_FOLD_STAGES: usize = 2;

/// Delay effect parameterized by persistence pairs.
///
/// Each persistence pair (birth, death) in the diagram maps to a delay tap:
///   delay_time = f(birth)        — when the feature appeared
///   feedback   = g(persistence)  — how long it lived
///   level      = h(dimension)    — which homological dimension
///
/// This creates a multi-tap delay where topological features literally
/// echo through time. Long-lived features (high persistence) echo longer.
#[derive(Debug, Clone)]
pub struct PersistenceDelay {
    buffer: Vec<f64>,
    write_pos: usize,
    taps: Vec<DelayTap>,
    sample_rate: f64,
    max_delay_samples: usize,
}

#[derive(Debug, Clone)]
struct DelayTap {
    delay_samples: usize,
    feedback: f64,
    level: f64,
}

impl PersistenceDelay {
    pub fn new(sample_rate: f64, max_delay_secs: f64) -> Self {
        let max_delay_samples = (sample_rate * max_delay_secs) as usize;
        Self {
            buffer: vec![0.0; max_delay_samples],
            write_pos: 0,
            taps: Vec::new(),
            sample_rate,
            max_delay_samples,
        }
    }

    /// Configure delay taps from persistence pairs.
    /// birth → delay time (scaled to [0, max_delay])
    /// persistence → feedback amount
    /// dimension → level scaling
    pub fn set_from_persistence(
        &mut self,
        pairs: &[(f64, f64, usize)], // (birth, death, dim)
        max_filtration: f64,
    ) {
        self.taps.clear();
        for &(birth, death, dim) in pairs {
            let persistence = if death.is_infinite() {
                max_filtration
            } else {
                death - birth
            };
            let delay_secs =
                (birth / max_filtration) * (self.max_delay_samples as f64 / self.sample_rate);
            let delay_samples = ((delay_secs * self.sample_rate) as usize)
                .min(self.max_delay_samples - 1)
                .max(1);
            let feedback = (persistence / max_filtration).clamp(0.0, 0.95);
            let level = 1.0 / (1.0 + dim as f64); // higher dimensions are quieter

            self.taps.push(DelayTap {
                delay_samples,
                feedback,
                level,
            });
        }

        // Every persistence pair adds a tap, and a point cloud of n points has n - 1 finite
        // dimension-0 pairs alone. Unscaled, the loop gain grows with the number of pairs.
        //
        // The wet path is out = x + sum_i level_i * y[n - d_i], and the loop is
        // y[n] = x[n] + sum_i g_i * y[n - d_i] with g_i = feedback_i * level_i. That loop is stable
        // when sum |g_i| < 1, and then |y| <= a / (1 - G) for an input bounded by a.
        // Scale so that sum level_i <= MAX_WET_LEVEL and G <= MAX_LOOP_GAIN. Uniform scaling keeps
        // the ratios: higher dimensions stay quieter and long-lived features still echo longest.
        let level_sum: f64 = self.taps.iter().map(|t| t.level).sum();
        if level_sum > MAX_WET_LEVEL {
            let scale = MAX_WET_LEVEL / level_sum;
            for tap in &mut self.taps {
                tap.level *= scale;
            }
        }
        let loop_gain: f64 = self.taps.iter().map(|t| t.feedback * t.level).sum();
        if loop_gain > MAX_LOOP_GAIN {
            let scale = MAX_LOOP_GAIN / loop_gain;
            for tap in &mut self.taps {
                tap.feedback *= scale;
            }
        }
    }

    pub fn process(&mut self, input: f64) -> f64 {
        let mut output = input;
        let mut fb_sum = input;

        // Single pass over taps: accumulate both output and feedback simultaneously.
        for tap in &self.taps {
            let read_pos = (self.write_pos + self.max_delay_samples - tap.delay_samples)
                % self.max_delay_samples;
            let delayed = self.buffer[read_pos];
            output += delayed * tap.level;
            fb_sum += delayed * tap.feedback * tap.level;
        }

        // The clamp on the write is the stability guard for a loud input. The wet output is not
        // clamped here: limiting belongs once, at the end of the chain, not in the middle of it.
        self.buffer[self.write_pos] = fb_sum.clamp(-1.0, 1.0);
        self.write_pos = (self.write_pos + 1) % self.max_delay_samples;
        output
    }

    pub fn process_buffer(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

/// Reverb effect driven by curvature of the underlying space.
///
/// In Riemannian geometry, curvature determines how geodesics spread:
/// - Positive curvature (sphere): geodesics converge → short, focused reverb
/// - Zero curvature (flat): geodesics stay parallel → medium reverb
/// - Negative curvature (hyperbolic): geodesics diverge → long, diffuse reverb
///
/// We use the Euler characteristic χ as a discrete curvature proxy via
/// the Gauss-Bonnet theorem: ∫ K dA = 2πχ.
///
/// Implementation: Schroeder reverb with allpass + comb filters,
/// where the number and spacing of combs is controlled by curvature.
#[derive(Debug, Clone)]
pub struct CurvatureReverb {
    comb_filters: Vec<CombFilter>,
    allpass_filters: Vec<AllpassFilter>,
    mix: f64,
}

#[derive(Debug, Clone)]
struct CombFilter {
    buffer: Vec<f64>,
    pos: usize,
    feedback: f64,
    damp: f64,
    damp_state: f64,
}

impl CombFilter {
    fn new(delay_samples: usize, feedback: f64, damp: f64) -> Self {
        Self {
            buffer: vec![0.0; delay_samples],
            pos: 0,
            feedback,
            damp,
            damp_state: 0.0,
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        let output = self.buffer[self.pos];
        self.damp_state = output * (1.0 - self.damp) + self.damp_state * self.damp;
        self.buffer[self.pos] = input + self.damp_state * self.feedback;
        self.pos = (self.pos + 1) % self.buffer.len();
        output
    }
}

#[derive(Debug, Clone)]
struct AllpassFilter {
    buffer: Vec<f64>,
    pos: usize,
    feedback: f64,
}

impl AllpassFilter {
    fn new(delay_samples: usize, feedback: f64) -> Self {
        Self {
            buffer: vec![0.0; delay_samples],
            pos: 0,
            feedback,
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        let buffered = self.buffer[self.pos];
        let output = -input + buffered;
        self.buffer[self.pos] = input + buffered * self.feedback;
        self.pos = (self.pos + 1) % self.buffer.len();
        output
    }
}

impl CurvatureReverb {
    /// Create a reverb parameterized by the Euler characteristic.
    /// χ > 0 → short, bright reverb (sphere-like)
    /// χ = 0 → medium reverb (torus-like)
    /// χ < 0 → long, dark reverb (high-genus surface)
    pub fn from_euler_characteristic(chi: isize, sample_rate: f64) -> Self {
        // Map curvature to reverb time: higher genus → longer reverb
        let rt60 = match chi {
            c if c > 0 => 0.5 + 0.3 / c as f64, // short
            0 => 1.5,                           // medium
            c => 1.5 + 0.5 * (-c as f64),       // long
        };

        let damping = match chi {
            c if c > 0 => 0.2, // bright
            0 => 0.5,
            _ => 0.8, // dark
        };

        // Comb filter delays in milliseconds (Schroeder/Freeverb standard values).
        // These are primes or coprime to each other, scaled to sample rate.
        // Using primes avoids common factors that would create periodic patterns.
        let comb_ms = [29.7, 37.1, 41.1, 43.7, 47.9, 53.0, 59.3, 61.7];
        let scale = sample_rate / 44100.0; // scale all delays relative to 44.1kHz reference
        let rt_scale = rt60 / 1.5; // scale relative to baseline reverb time

        let feedback = (-6.908 * comb_ms[0] / 1000.0 / rt60).exp(); // -60dB at rt60

        let comb_filters = comb_ms
            .iter()
            .map(|&ms| {
                let delay = ((ms / 1000.0) * sample_rate * rt_scale).round() as usize;
                CombFilter::new(delay.max(1), feedback, damping)
            })
            .collect();

        // Allpass delays also scaled to sample rate.
        // Standard Schroeder values: ~5ms, ~1.7ms, ~1ms
        let allpass_filters = vec![
            AllpassFilter::new(((5.0 / 1000.0) * sample_rate * scale).round() as usize, 0.5),
            AllpassFilter::new(((1.7 / 1000.0) * sample_rate * scale).round() as usize, 0.5),
            AllpassFilter::new(((1.0 / 1000.0) * sample_rate * scale).round() as usize, 0.5),
        ];

        Self {
            comb_filters,
            allpass_filters,
            mix: 0.3,
        }
    }

    pub fn set_mix(&mut self, mix: f64) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    pub fn process(&mut self, input: f64) -> f64 {
        // Sum comb filter outputs (parallel)
        let comb_sum: f64 = self.comb_filters.iter_mut().map(|c| c.process(input)).sum();
        let comb_out = comb_sum / self.comb_filters.len() as f64;

        // Chain through allpass filters (series)
        let mut out = comb_out;
        for ap in &mut self.allpass_filters {
            out = ap.process(out);
        }

        // Wet/dry mix
        input * (1.0 - self.mix) + out * self.mix
    }

    pub fn process_buffer(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

/// Distortion/saturation driven by Betti numbers.
///
/// The idea: β₀ (components) controls the number of wavefold stages,
/// β₁ (loops) controls the feedback amount in the wavefolder,
/// β₂ (voids) controls the asymmetry of the transfer curve.
///
/// Waveshaping is fundamentally a nonlinear map ℝ → ℝ.
/// The topology of the graph of this map determines the harmonic content.
#[derive(Debug, Clone)]
pub struct BettiDistortion {
    fold_stages: usize,
    feedback: f64,
    asymmetry: f64,
    drive: f64,
    prev_sample: f64,
}

impl BettiDistortion {
    pub fn from_betti(b0: usize, b1: usize, b2: usize) -> Self {
        Self {
            fold_stages: b0.clamp(1, MAX_FOLD_STAGES),
            feedback: (b1 as f64 * 0.15).min(0.9),
            asymmetry: b2 as f64 * 0.1,
            drive: 1.0,
            prev_sample: 0.0,
        }
    }

    pub fn set_drive(&mut self, drive: f64) {
        self.drive = drive.max(0.1);
    }

    pub fn process(&mut self, input: f64) -> f64 {
        let driven = (input + self.prev_sample * self.feedback) * self.drive;

        // Multi-stage wavefolding
        let mut x = driven;
        for _ in 0..self.fold_stages {
            // Sine wavefolder: maps ℝ → [-1, 1] with wrapping
            x = (x * std::f64::consts::FRAC_PI_2).sin();
        }

        // Asymmetric saturation based on β₂
        if self.asymmetry > 0.0 {
            x = if x >= 0.0 {
                x.tanh()
            } else {
                (x * (1.0 + self.asymmetry)).tanh() / (1.0 + self.asymmetry)
            };
        }

        self.prev_sample = x;
        x
    }

    pub fn process_buffer(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_doesnt_clip() {
        let mut delay = PersistenceDelay::new(44100.0, 2.0);
        delay.set_from_persistence(&[(0.0, 0.5, 0), (0.2, 0.8, 1)], 1.0);
        for _ in 0..1000 {
            let out = delay.process(0.5);
            assert!(out.abs() <= 1.0);
        }
    }

    /// Many persistence pairs, as a 10-point circle produces. The delay is the feedback network
    /// y[n] = x[n] + sum_i g_i y[n - d_i], which is stable only if sum |g_i| < 1. The bound below follows
    /// from that: |y| <= a / (1 - G), and the wet output adds at most sum(level) * |y|.
    /// The internal clamp(-1, 1) must never be what keeps the output in range, so use a quiet input.
    fn many_pairs() -> Vec<(f64, f64, usize)> {
        let mut pairs: Vec<(f64, f64, usize)> =
            (1..=40).map(|i| (0.0, 0.02 * i as f64 + 0.1, 0)).collect();
        pairs.extend((1..=10).map(|i| (0.05 * i as f64, 0.05 * i as f64 + 0.4, 1)));
        pairs.push((0.0, f64::INFINITY, 0));
        pairs
    }

    #[test]
    fn delay_loop_gain_bounded_for_sine() {
        let sr = 48000.0;
        let mut delay = PersistenceDelay::new(sr, 2.0);
        delay.set_from_persistence(&many_pairs(), 1.0);
        let a = 0.2;
        let mut peak: f64 = 0.0;
        for n in 0..(5 * 48000) {
            let x = a * (std::f64::consts::TAU * 220.0 * n as f64 / sr).sin();
            peak = peak.max(delay.process(x).abs());
        }
        let bound = a * (1.0 + MAX_WET_LEVEL / (1.0 - MAX_LOOP_GAIN));
        assert!(
            peak < bound + 1e-9,
            "delay peak {peak} exceeds the bound {bound}: feedback is unstable"
        );
    }

    #[test]
    fn delay_loop_gain_bounded_for_dc() {
        let mut delay = PersistenceDelay::new(48000.0, 2.0);
        delay.set_from_persistence(&many_pairs(), 1.0);
        let mut peak: f64 = 0.0;
        for _ in 0..(5 * 48000) {
            peak = peak.max(delay.process(0.2).abs());
        }
        let bound = 0.2 * (1.0 + MAX_WET_LEVEL / (1.0 - MAX_LOOP_GAIN));
        assert!(
            peak < bound + 1e-9,
            "delay peak {peak} exceeds the bound {bound}: feedback is unstable"
        );
    }

    #[test]
    fn reverb_stability() {
        let mut reverb = CurvatureReverb::from_euler_characteristic(2, 44100.0);
        for _ in 0..44100 {
            let out = reverb.process(0.0);
            assert!(out.abs() < 10.0); // shouldn't blow up
        }
    }

    /// Each sine fold has small-signal gain pi/2, so k stages multiply to (pi/2)^k: x1.6 for one
    /// component, x9.5 for five, x91 for ten. A quiet signal must not turn into fuzz because the
    /// cloud has several components.
    #[test]
    fn distortion_small_signal_gain_is_bounded() {
        let a = 0.05;
        let limit = (std::f64::consts::FRAC_PI_2).powi(MAX_FOLD_STAGES as i32) * 1.01;
        for b0 in [1, 2, 5, 10, 50] {
            let mut dist = BettiDistortion::from_betti(b0, 0, 0);
            let mut peak: f64 = 0.0;
            for n in 0..4800 {
                let x = a * (std::f64::consts::TAU * 220.0 * n as f64 / 48000.0).sin();
                peak = peak.max(dist.process(x).abs());
            }
            assert!(
                peak / a <= limit,
                "b0 = {b0}: small-signal gain {:.1} exceeds {:.2}",
                peak / a,
                limit
            );
        }
    }

    #[test]
    fn distortion_bounded() {
        let mut dist = BettiDistortion::from_betti(3, 2, 1);
        dist.set_drive(4.0);
        for _ in 0..1000 {
            let out = dist.process(0.8);
            assert!(out.abs() <= 1.01);
        }
    }
}
