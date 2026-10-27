use crate::audio::filter::{FilterMode, TopologicalFilter};
use crate::audio::oscillator::{Oscillator, Waveform};
use crate::topology::homology::BettiNumbers;

/// BettiDrone: a drone synthesizer whose harmonic structure is determined
/// by the Betti numbers of a simplicial complex.
///
/// ## Design
///
/// - **β₀ (connected components)** → number of fundamental oscillators.
///   Each component gets its own pitch with slight detuning for richness.
///
/// - **β₁ (loops)** → number of modulation oscillators (LFOs).
///   Each independent loop becomes a cyclic modulation source.
///   Modulation is applied additively to the output rather than by
///   mutating oscillator frequencies (which causes numerical drift).
///
/// - **β₂ (voids)** → number of sub-oscillators (one octave below).
///   Enclosed voids add depth, like the resonant cavity of an instrument.
///
/// ## Tuning
///
/// The f-vector ratios f₁/f₀, f₂/f₁, etc. become just-intonation
/// frequency ratios between oscillators.
pub struct BettiDrone {
    /// Fundamental oscillators — one per connected component (β₀)
    fundamentals: Vec<Oscillator>,
    /// LFO modulators — one per independent loop (β₁)
    modulators: Vec<Oscillator>,
    /// Sub-oscillators — one per void (β₂)
    subs: Vec<Oscillator>,
    /// Output filter
    filter: TopologicalFilter,
    /// Amplitude envelope
    amplitude: f64,
    /// Base frequency
    base_freq: f64,
}

impl BettiDrone {
    pub fn new(betti: &BettiNumbers, base_freq: f64, sample_rate: f64) -> Self {
        let b0 = betti.b0().max(1);
        let b1 = betti.b1();
        let b2 = betti.b2();

        // Create fundamental oscillators with slight detuning for richness
        let fundamentals: Vec<Oscillator> = (0..b0)
            .map(|i| {
                let detune = 1.0 + (i as f64 * 0.002);
                Oscillator::new(Waveform::Sawtooth, base_freq * detune, sample_rate)
            })
            .collect();

        // LFO modulators at sub-audio rates
        let modulators: Vec<Oscillator> = (0..b1)
            .map(|i| {
                let rate = 0.1 + i as f64 * 0.07; // 0.1 Hz, 0.17 Hz, 0.24 Hz...
                let mut osc = Oscillator::new(Waveform::Sine, rate, sample_rate);
                osc.amplitude = 0.15 * (1.0 + i as f64 * 0.5); // modulation depth
                osc
            })
            .collect();

        // Sub-oscillators one octave below
        let subs: Vec<Oscillator> = (0..b2)
            .map(|i| {
                let sub_freq = base_freq / 2.0 * (1.0 + i as f64 * 0.01);
                let mut osc = Oscillator::new(Waveform::Sine, sub_freq, sample_rate);
                osc.amplitude = 0.6;
                osc
            })
            .collect();

        let filter = TopologicalFilter::new(FilterMode::LowPass, base_freq * 8.0, 1.5, sample_rate);

        Self {
            fundamentals,
            modulators,
            subs,
            filter,
            amplitude: 0.3,
            base_freq,
        }
    }

    /// Set the just-intonation ratios from the f-vector of the complex.
    pub fn set_ratios_from_f_vector(&mut self, f_vector: &[usize]) {
        let mut ratios = vec![1.0_f64];
        for window in f_vector.windows(2) {
            if window[0] > 0 {
                ratios.push(window[1] as f64 / window[0] as f64);
            }
        }

        // Retune fundamentals according to ratios
        for (i, osc) in self.fundamentals.iter_mut().enumerate() {
            let ratio = ratios.get(i % ratios.len()).copied().unwrap_or(1.0);
            osc.set_frequency(self.base_freq * ratio);
        }
    }

    /// Generate the next sample.
    ///
    /// Modulation is applied as amplitude modulation (ring mod) on the mixed
    /// output, not as FM on individual oscillators. This avoids the numerical
    /// instability of set_frequency/restore_frequency every sample, and
    /// produces a cleaner modulation spectrum.
    pub fn next_sample(&mut self) -> f64 {
        // Sum fundamentals
        let fund_sum: f64 = self.fundamentals.iter_mut().map(|o| o.next_sample()).sum();
        let fund_level = if self.fundamentals.is_empty() {
            0.0
        } else {
            fund_sum / self.fundamentals.len() as f64
        };

        // Sum sub-oscillators
        let sub_sum: f64 = self.subs.iter_mut().map(|o| o.next_sample()).sum();
        let sub_level = if self.subs.is_empty() {
            0.0
        } else {
            sub_sum / self.subs.len() as f64
        };

        let mixed = fund_level + sub_level * 0.5;

        // Apply LFO modulation as amplitude modulation: output * (1 + lfo_sum)
        let lfo_sum: f64 = self.modulators.iter_mut().map(|m| m.next_sample()).sum();
        let modulated = mixed * (1.0 + lfo_sum);

        self.filter.process(modulated) * self.amplitude
    }

    pub fn fill_buffer(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.next_sample();
        }
    }

    pub fn set_amplitude(&mut self, amp: f64) {
        self.amplitude = amp.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drone_produces_sound() {
        let betti = BettiNumbers {
            values: vec![1, 2, 1],
        };
        let mut drone = BettiDrone::new(&betti, 110.0, 44100.0);
        let mut buffer = vec![0.0; 1024];
        drone.fill_buffer(&mut buffer);
        let energy: f64 = buffer.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }
}
