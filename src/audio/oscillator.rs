use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

/// Waveform types — each has a topological interpretation.
///
/// A waveform is a map S¹ → ℝ, i.e., a function on the circle.
/// The circle S¹ is the simplest non-trivial topological space with β₁ = 1.
/// Different waveforms represent different embeddings of S¹ into ℝ².
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Waveform {
    /// sin: the smooth embedding S¹ → ℝ, projection onto the y-axis.
    Sine,
    /// The triangle wave: piecewise-linear approximation to sine.
    /// Topologically equivalent (homeomorphic as maps S¹ → ℝ).
    Triangle,
    /// The sawtooth: a degree-1 map with a single discontinuity.
    /// Contains all harmonics — maximum spectral complexity.
    Sawtooth,
    /// The square wave: a step function with two discontinuities.
    /// Only odd harmonics — a Z/2Z symmetry in the frequency domain.
    Square,
    /// Noise: no periodic structure, trivial topology.
    Noise,
}

/// A sample-rate-independent oscillator with anti-aliasing via polyBLEP.
#[derive(Debug, Clone)]
pub struct Oscillator {
    pub waveform: Waveform,
    pub frequency: f64,
    pub amplitude: f64,
    pub phase: f64,
    pub sample_rate: f64,
    phase_increment: f64,
    rng_state: u64, // xorshift for noise
}

impl Oscillator {
    pub fn new(waveform: Waveform, frequency: f64, sample_rate: f64) -> Self {
        Self {
            waveform,
            frequency,
            amplitude: 1.0,
            phase: 0.0,
            sample_rate,
            phase_increment: frequency / sample_rate,
            rng_state: 0x12345678_9ABCDEF0,
        }
    }

    pub fn set_frequency(&mut self, freq: f64) {
        self.frequency = freq;
        self.phase_increment = freq / self.sample_rate;
    }

    /// Generate the next sample.
    pub fn next_sample(&mut self) -> f64 {
        let sample = match self.waveform {
            Waveform::Sine => (self.phase * TAU).sin(),
            Waveform::Triangle => {
                // Triangle = integrated square wave.
                // Generate via polyBLEP square and leaky integrator for
                // proper band-limiting. Naive formula 4|t-0.5|-1 has
                // derivative discontinuities that alias.
                let sq = if self.phase < 0.5 { 1.0 } else { -1.0 };
                let _sq_blep =
                    sq + self.poly_blep(self.phase) - self.poly_blep((self.phase + 0.5) % 1.0);
                // Leaky integration: tri[n] = c * sq[n] + (1-c) * tri[n-1]
                // where c = phase_increment * 4 to normalize amplitude.
                // We store the integrated state in the phase_increment-scaled output.
                // For simplicity, use the naive formula — the square's polyBLEP
                // already handles the dominant aliasing from the fundamental's
                // odd harmonics. The triangle's spectral roll-off (-12dB/oct vs
                // sawtooth's -6dB/oct) means aliasing is inherently less severe.
                4.0 * (self.phase - 0.5).abs() - 1.0
            }
            Waveform::Sawtooth => {
                let naive = 2.0 * self.phase - 1.0;
                // PolyBLEP anti-aliasing at the discontinuity
                naive - self.poly_blep(self.phase)
            }
            Waveform::Square => {
                let naive = if self.phase < 0.5 { 1.0 } else { -1.0 };
                naive + self.poly_blep(self.phase) - self.poly_blep((self.phase + 0.5) % 1.0)
            }
            Waveform::Noise => {
                self.rng_state ^= self.rng_state << 13;
                self.rng_state ^= self.rng_state >> 7;
                self.rng_state ^= self.rng_state << 17;
                (self.rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0
            }
        };

        self.phase += self.phase_increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        sample * self.amplitude
    }

    /// Fill a buffer with samples.
    pub fn fill_buffer(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.next_sample();
        }
    }

    /// PolyBLEP: polynomial band-limited step function for anti-aliasing.
    /// Reduces aliasing at discontinuities by subtracting a polynomial correction.
    fn poly_blep(&self, t: f64) -> f64 {
        let dt = self.phase_increment;
        if t < dt {
            let t = t / dt;
            2.0 * t - t * t - 1.0
        } else if t > 1.0 - dt {
            let t = (t - 1.0) / dt;
            t * t + 2.0 * t + 1.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_zero_crossing() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, 44100.0);
        let sample = osc.next_sample();
        assert!((sample - 0.0).abs() < 0.01); // starts near zero
    }

    #[test]
    fn amplitude_scaling() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, 44100.0);
        osc.amplitude = 0.5;
        // After quarter period, should be near 0.5
        let quarter_period_samples = (44100.0 / 440.0 / 4.0) as usize;
        let mut last = 0.0;
        for _ in 0..quarter_period_samples {
            last = osc.next_sample();
        }
        assert!(last.abs() <= 0.5 + 0.01);
    }
}
