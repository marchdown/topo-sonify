use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

/// Filter mode — topologically motivated.
///
/// A filter is a projection operator on the signal space L²([0,T]).
/// Different modes correspond to different subspaces:
/// - LowPass: project onto the span of low-frequency eigenfunctions of the Laplacian
/// - HighPass: project onto the complement
/// - BandPass: project onto a specific eigenvalue band
/// - Notch: remove a specific eigenvalue band
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterMode {
    LowPass,
    HighPass,
    BandPass,
    Notch,
}

/// A biquad filter whose parameters can be driven by topological invariants.
///
/// The transfer function H(z) = (b₀ + b₁z⁻¹ + b₂z⁻²) / (1 + a₁z⁻¹ + a₂z⁻²)
/// lives in the space of rational functions on the unit circle S¹ ⊂ ℂ.
/// Poles inside the unit disk → stable; the topology of the pole/zero
/// configuration determines the filter's character.
#[derive(Debug, Clone)]
pub struct TopologicalFilter {
    pub mode: FilterMode,
    pub cutoff: f64,
    pub resonance: f64, // Q factor
    pub sample_rate: f64,

    // Biquad coefficients
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,

    // State
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl TopologicalFilter {
    pub fn new(mode: FilterMode, cutoff: f64, resonance: f64, sample_rate: f64) -> Self {
        let mut filter = Self {
            mode,
            cutoff,
            resonance,
            sample_rate,
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        };
        filter.compute_coefficients();
        filter
    }

    /// Recompute biquad coefficients from cutoff and resonance.
    /// This is the "chart map" from parameter space (cutoff, Q) to
    /// coefficient space (b₀, b₁, b₂, a₁, a₂) — a smooth map between manifolds.
    pub fn compute_coefficients(&mut self) {
        let omega = TAU * self.cutoff / self.sample_rate;
        let sin_w = omega.sin();
        let cos_w = omega.cos();
        let alpha = sin_w / (2.0 * self.resonance);

        match self.mode {
            FilterMode::LowPass => {
                let b1 = 1.0 - cos_w;
                self.b0 = b1 / 2.0;
                self.b1 = b1;
                self.b2 = b1 / 2.0;
                self.a1 = -2.0 * cos_w;
                self.a2 = 1.0 - alpha;
            }
            FilterMode::HighPass => {
                let b1 = 1.0 + cos_w;
                self.b0 = b1 / 2.0;
                self.b1 = -(1.0 + cos_w);
                self.b2 = b1 / 2.0;
                self.a1 = -2.0 * cos_w;
                self.a2 = 1.0 - alpha;
            }
            FilterMode::BandPass => {
                self.b0 = alpha;
                self.b1 = 0.0;
                self.b2 = -alpha;
                self.a1 = -2.0 * cos_w;
                self.a2 = 1.0 - alpha;
            }
            FilterMode::Notch => {
                self.b0 = 1.0;
                self.b1 = -2.0 * cos_w;
                self.b2 = 1.0;
                self.a1 = -2.0 * cos_w;
                self.a2 = 1.0 - alpha;
            }
        }

        // Normalize
        let a0 = 1.0 + alpha;
        self.b0 /= a0;
        self.b1 /= a0;
        self.b2 /= a0;
        self.a1 /= a0;
        self.a2 /= a0;
    }

    /// Set cutoff frequency and recompute.
    pub fn set_cutoff(&mut self, cutoff: f64) {
        self.cutoff = cutoff.clamp(20.0, self.sample_rate / 2.0 - 1.0);
        self.compute_coefficients();
    }

    /// Set resonance (Q factor) and recompute.
    pub fn set_resonance(&mut self, q: f64) {
        self.resonance = q.max(0.1);
        self.compute_coefficients();
    }

    /// Process a single sample through the filter.
    pub fn process(&mut self, input: f64) -> f64 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// Process a buffer in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }

    /// Reset filter state (clear delay line).
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowpass_attenuates_high_freq() {
        let mut filter = TopologicalFilter::new(FilterMode::LowPass, 200.0, 0.707, 44100.0);
        // Feed in a high frequency signal
        let mut energy = 0.0;
        for i in 0..1000 {
            let input = (i as f64 * TAU * 10000.0 / 44100.0).sin();
            let output = filter.process(input);
            energy += output * output;
        }
        // Should be heavily attenuated
        assert!(energy < 100.0);
    }
}
