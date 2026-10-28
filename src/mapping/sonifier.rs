use crate::audio::effects::{BettiDistortion, CurvatureReverb, PersistenceDelay};
use crate::instruments::betti_drone::BettiDrone;
use crate::instruments::morse_synth::MorseSynth;
use crate::instruments::simplicial_pluck::SimplicialPluck;
use crate::topology::complex::SimplicialComplex;
use crate::topology::filtration::FilteredComplex;
use crate::topology::hodge::HodgeLaplacian;
use crate::topology::homology::BettiNumbers;
use crate::topology::persistence::PersistenceDiagram;
use crate::topology::standard_persistence::compute_persistence_twist;

/// The central sonification engine.
///
/// TopologySonifier takes a simplicial complex (or point cloud) and
/// wires up the complete signal chain:
///
///   Topology → Invariants → Instruments → Effects → Output
///
/// All topology computation flows through the standard persistence
/// algorithm (ELZ 2002 + twist optimization), the Hodge Laplacian,
/// and Betti numbers. Nothing is approximated.
pub struct TopologySonifier {
    // Topology
    pub complex: SimplicialComplex,
    pub betti: BettiNumbers,
    pub persistence: Option<PersistenceDiagram>,
    pub hodge_eigenvalues: Vec<Vec<f64>>,
    pub spectral_gap: f64,

    // Instruments
    pub drone: BettiDrone,
    pub pluck: SimplicialPluck,
    pub morse: MorseSynth,

    // Effects
    pub delay: PersistenceDelay,
    pub reverb: CurvatureReverb,
    pub distortion: BettiDistortion,

    // Mixing
    pub drone_level: f64,
    pub pluck_level: f64,
    pub morse_level: f64,
    pub master_volume: f64,
}

impl TopologySonifier {
    /// Create a sonifier from a simplicial complex.
    pub fn from_complex(complex: SimplicialComplex, base_freq: f64, sample_rate: f64) -> Self {
        let betti = BettiNumbers::compute(&complex);
        let chi = complex.euler_characteristic();
        let f_vec = complex.f_vector();

        // Hodge Laplacian eigenvalues for each dimension.
        let mut hodge_eigenvalues = Vec::new();
        let mut spectral_gap = 0.0;
        for dim in 0..=complex.dimension() as usize {
            if complex.count_dim(dim as isize) > 0 {
                let l = HodgeLaplacian::compute(&complex, dim);
                if dim == 0 {
                    spectral_gap = l.spectral_gap();
                }
                hodge_eigenvalues.push(l.eigenvalues());
            }
        }

        // Build instruments
        let mut drone = BettiDrone::new(&betti, base_freq, sample_rate);
        drone.set_ratios_from_f_vector(&f_vec);

        let mut pluck = SimplicialPluck::new(base_freq, sample_rate);
        pluck.configure(base_freq, &f_vec, betti.b0(), betti.b1());

        let morse = MorseSynth::new(sample_rate);

        // Build effects
        let delay = PersistenceDelay::new(sample_rate, 2.0);
        let reverb = CurvatureReverb::from_euler_characteristic(chi, sample_rate);
        let distortion = BettiDistortion::from_betti(betti.b0(), betti.b1(), betti.b2());

        Self {
            complex,
            betti,
            persistence: None,
            hodge_eigenvalues,
            spectral_gap,
            drone,
            pluck,
            morse,
            delay,
            reverb,
            distortion,
            drone_level: 0.5,
            pluck_level: 0.3,
            morse_level: 0.4,
            master_volume: 0.7,
        }
    }

    /// Create from a point cloud — builds a Vietoris-Rips complex,
    /// computes persistence via the standard algorithm (not Betti tracking),
    /// and wires everything through.
    pub fn from_point_cloud(
        points: &[[f64; 2]],
        radius: f64,
        base_freq: f64,
        sample_rate: f64,
    ) -> Self {
        let complex = SimplicialComplex::vietoris_rips(points, radius);
        let mut sonifier = Self::from_complex(complex, base_freq, sample_rate);

        // Compute persistence via the real algorithm: column reduction on the
        // filtration boundary matrix with twist optimization.
        let fc = FilteredComplex::vietoris_rips(points, radius);
        let pd = compute_persistence_twist(&fc);

        // Wire persistence into delay taps.
        let pairs: Vec<(f64, f64, usize)> = pd
            .pairs
            .iter()
            .map(|p| (p.birth, p.death, p.dimension))
            .collect();
        let max_filt = pd.max_persistence().max(1.0);
        sonifier.delay.set_from_persistence(&pairs, max_filt);

        // Wire persistence into Morse synth.
        sonifier.morse = MorseSynth::from_persistence_pairs(
            &pairs,
            (base_freq / 2.0, base_freq * 4.0),
            2.0,
            sample_rate,
        );

        sonifier.persistence = Some(pd);
        sonifier
    }

    /// Trigger the pluck instrument.
    pub fn pluck_trigger(&mut self) {
        let f_vec = self.complex.f_vector();
        self.pluck.pluck(&f_vec);
    }

    /// Generate the next audio sample through the full signal chain.
    pub fn next_sample(&mut self) -> f64 {
        let drone_out = self.drone.next_sample() * self.drone_level;
        let pluck_out = self.pluck.next_sample() * self.pluck_level;
        let morse_out = self.morse.next_sample() * self.morse_level;

        let dry = drone_out + pluck_out + morse_out;
        let distorted = self.distortion.process(dry);
        let delayed = self.delay.process(distorted);
        let reverbed = self.reverb.process(delayed);

        // The only limiter in the chain. A rich effects chain has peaks above full scale; a hard
        // clamp would flatten them into square edges, tanh rounds them off (and is within 5% of
        // linear below 0.4). Output stays strictly inside (-1, 1).
        (reverbed * self.master_volume).tanh()
    }

    /// Fill a buffer with audio.
    pub fn fill_buffer(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.next_sample();
        }
    }

    /// Get a JSON-serializable snapshot of the current topological state.
    pub fn topology_state(&self) -> TopologyState {
        TopologyState {
            f_vector: self.complex.f_vector(),
            betti_numbers: self.betti.values.clone(),
            euler_characteristic: self.complex.euler_characteristic(),
            dimension: self.complex.dimension(),
            num_simplices: self.complex.size(),
            spectral_gap: self.spectral_gap,
            persistence_pairs: self
                .persistence
                .as_ref()
                .map(|pd| {
                    pd.pairs
                        .iter()
                        .map(|p| PersistencePairState {
                            birth: p.birth,
                            death: if p.is_essential() { -1.0 } else { p.death },
                            dimension: p.dimension,
                            persistence: if p.is_essential() {
                                -1.0
                            } else {
                                p.persistence()
                            },
                        })
                        .collect()
                })
                .unwrap_or_default(),
            persistence_entropy: self.persistence.as_ref().map(|p| p.persistence_entropy()),
            total_persistence: self.persistence.as_ref().map(|p| p.total_persistence()),
            hodge_eigenvalues: self.hodge_eigenvalues.clone(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TopologyState {
    pub f_vector: Vec<usize>,
    pub betti_numbers: Vec<usize>,
    pub euler_characteristic: isize,
    pub dimension: isize,
    pub num_simplices: usize,
    pub spectral_gap: f64,
    pub persistence_pairs: Vec<PersistencePairState>,
    pub persistence_entropy: Option<f64>,
    pub total_persistence: Option<f64>,
    pub hodge_eigenvalues: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersistencePairState {
    pub birth: f64,
    pub death: f64, // -1 for essential
    pub dimension: usize,
    pub persistence: f64, // -1 for essential
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sonifier_from_complex() {
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1, 2]);
        k.insert_simplex(&[1, 2, 3]);
        let mut sonifier = TopologySonifier::from_complex(k, 220.0, 44100.0);
        let mut buffer = vec![0.0; 4096];
        sonifier.fill_buffer(&mut buffer);
        let energy: f64 = buffer.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn sonifier_from_point_cloud() {
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [2.0, 0.5]];
        let mut sonifier = TopologySonifier::from_point_cloud(&points, 1.2, 220.0, 44100.0);
        sonifier.pluck_trigger();
        let mut buffer = vec![0.0; 4096];
        sonifier.fill_buffer(&mut buffer);
        let state = sonifier.topology_state();
        assert!(!state.betti_numbers.is_empty());
        // Verify persistence was computed via the real algorithm.
        assert!(!state.persistence_pairs.is_empty());
        // Verify Hodge eigenvalues were computed.
        assert!(!state.hodge_eigenvalues.is_empty());
    }

    /// Render three seconds with the default mixer and check the result is a usable signal and
    /// not a saturated one. Regression for runaway delay feedback: the loop gain grew with the
    /// number of persistence pairs, so a 10-point circle came out as a full-scale square wave
    /// (RMS 1.0), worse with every point added.
    fn assert_usable_output(points: &[[f64; 2]], radius: f64) {
        let sr = 48000.0;
        let mut sonifier = TopologySonifier::from_point_cloud(points, radius, 220.0, sr);
        let mut seconds = Vec::new();
        for _ in 0..3 {
            let mut buf = vec![0.0; sr as usize];
            sonifier.fill_buffer(&mut buf);
            assert!(buf.iter().all(|x| x.is_finite()), "non-finite sample");
            let flat = buf.iter().filter(|x| x.abs() >= 0.999).count();
            assert_eq!(
                flat, 0,
                "{flat} flat-topped samples: the output is saturating"
            );
            seconds.push((buf.iter().map(|x| x * x).sum::<f64>() / buf.len() as f64).sqrt());
        }
        for (i, rms) in seconds.iter().enumerate() {
            assert!(
                // Broken output measured RMS 0.95 or more; working output measures 0.4 to 0.6.
                (0.02..0.7).contains(rms),
                "second {i}: rms {rms:.3} is silent or saturated"
            );
        }
        assert!(
            seconds[2] <= 1.5 * seconds[0],
            "output grows over time: rms {:?}",
            seconds
        );
    }

    #[test]
    fn circle_preset_output_is_usable() {
        // The page's circle preset: 10 points on a circle of radius 0.3, slider radius 0.5.
        let points: Vec<[f64; 2]> = (0..10)
            .map(|i| {
                let a = i as f64 / 10.0 * std::f64::consts::TAU;
                [0.5 + 0.3 * a.cos(), 0.5 + 0.3 * a.sin()]
            })
            .collect();
        assert_usable_output(&points, 0.5);
    }

    #[test]
    fn larger_cloud_output_is_usable() {
        // 30 deterministic pseudo-random points: more persistence pairs, so more delay taps.
        let mut state: u64 = 12345;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let points: Vec<[f64; 2]> = (0..30).map(|_| [next(), next()]).collect();
        assert_usable_output(&points, 0.2);
    }

    #[test]
    fn topology_state_has_spectral_gap() {
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1, 2]);
        let sonifier = TopologySonifier::from_complex(k, 220.0, 44100.0);
        let state = sonifier.topology_state();
        assert!(state.spectral_gap > 0.0); // connected → positive gap
    }
}
