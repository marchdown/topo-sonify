// Range loops on matrix/vertex indices are idiomatic for numerical code —
// the index variable is used in arithmetic, not just for array access.
#![allow(clippy::needless_range_loop)]

pub mod audio;
pub mod instruments;
pub mod mapping;
pub mod topology;

// Re-export the full public API.
pub use mapping::TopologySonifier;
pub use topology::{
    bottleneck_distance, compute_persistence, compute_persistence_twist, wasserstein_distance,
    BettiNumbers, DiscreteMorseFunction, FilteredComplex, HodgeLaplacian, PersistenceDiagram,
    PersistencePair, Simplex, SimplicialComplex,
};

use wasm_bindgen::prelude::*;

// ─── WASM Interface ─────────────────────────────────────────────────

/// WASM-exposed sonification engine.
///
/// Bridges the Rust topology+audio engine to JavaScript/Web Audio.
/// All topology computation (persistence, Hodge, Betti) happens in Rust;
/// the browser handles audio output and UI.
#[wasm_bindgen]
pub struct WasmSonifier {
    inner: TopologySonifier,
}

#[wasm_bindgen]
impl WasmSonifier {
    /// Create a sonifier from a point cloud.
    /// Points are passed as a flat f64 array: [x0, y0, x1, y1, ...].
    /// This computes VR complex, standard persistence, Hodge eigenvalues,
    /// and wires everything into the audio engine.
    #[wasm_bindgen(constructor)]
    pub fn new(points_flat: &[f64], radius: f64, base_freq: f64, sample_rate: f64) -> Self {
        let points: Vec<[f64; 2]> = points_flat.chunks(2).map(|c| [c[0], c[1]]).collect();
        Self {
            inner: TopologySonifier::from_point_cloud(&points, radius, base_freq, sample_rate),
        }
    }

    /// Create from an explicit simplex list.
    #[wasm_bindgen]
    pub fn from_simplices(
        simplices_flat: &[usize],
        simplex_lengths: &[usize],
        base_freq: f64,
        sample_rate: f64,
    ) -> Self {
        let mut complex = SimplicialComplex::new();
        let mut offset = 0;
        for &len in simplex_lengths {
            let verts = &simplices_flat[offset..offset + len];
            complex.insert_simplex(verts);
            offset += len;
        }
        Self {
            inner: TopologySonifier::from_complex(complex, base_freq, sample_rate),
        }
    }

    /// Fill a Float64Array with audio samples.
    #[wasm_bindgen]
    pub fn fill_buffer(&mut self, buffer: &mut [f64]) {
        self.inner.fill_buffer(buffer);
    }

    /// Trigger the pluck instrument.
    #[wasm_bindgen]
    pub fn pluck(&mut self) {
        self.inner.pluck_trigger();
    }

    /// Get topological state as JSON — includes Betti, Euler, f-vector,
    /// persistence pairs, Hodge eigenvalues, spectral gap, entropy.
    #[wasm_bindgen]
    pub fn topology_state_json(&self) -> String {
        serde_json::to_string(&self.inner.topology_state()).unwrap_or_default()
    }

    /// Set mixer levels.
    #[wasm_bindgen]
    pub fn set_levels(&mut self, drone: f64, pluck: f64, morse: f64, master: f64) {
        self.inner.drone_level = drone;
        self.inner.pluck_level = pluck;
        self.inner.morse_level = morse;
        self.inner.master_volume = master;
    }

    #[wasm_bindgen]
    pub fn betti_numbers(&self) -> Vec<usize> {
        self.inner.betti.values.clone()
    }

    #[wasm_bindgen]
    pub fn euler_characteristic(&self) -> isize {
        self.inner.complex.euler_characteristic()
    }

    #[wasm_bindgen]
    pub fn f_vector(&self) -> Vec<usize> {
        self.inner.complex.f_vector()
    }

    #[wasm_bindgen]
    pub fn spectral_gap(&self) -> f64 {
        self.inner.spectral_gap
    }
}

/// WASM-exposed standalone topology computation (no audio).
/// For the interactive docs / exercises.
#[wasm_bindgen]
pub struct WasmTopology;

#[wasm_bindgen]
impl WasmTopology {
    /// Compute Betti numbers of a VR complex from a point cloud.
    #[wasm_bindgen]
    pub fn betti_from_points(points_flat: &[f64], radius: f64) -> Vec<usize> {
        let points: Vec<[f64; 2]> = points_flat.chunks(2).map(|c| [c[0], c[1]]).collect();
        let complex = SimplicialComplex::vietoris_rips(&points, radius);
        BettiNumbers::compute(&complex).values
    }

    /// Compute full persistence diagram from a point cloud. Returns JSON.
    #[wasm_bindgen]
    pub fn persistence_from_points(points_flat: &[f64], max_radius: f64) -> String {
        let points: Vec<[f64; 2]> = points_flat.chunks(2).map(|c| [c[0], c[1]]).collect();
        let fc = FilteredComplex::vietoris_rips(&points, max_radius);
        let pd = compute_persistence_twist(&fc);
        serde_json::to_string(&pd.pairs).unwrap_or_default()
    }

    /// Compute Hodge Laplacian eigenvalues. Returns JSON array of arrays.
    #[wasm_bindgen]
    pub fn hodge_eigenvalues(points_flat: &[f64], radius: f64) -> String {
        let points: Vec<[f64; 2]> = points_flat.chunks(2).map(|c| [c[0], c[1]]).collect();
        let complex = SimplicialComplex::vietoris_rips(&points, radius);
        let mut all_evs: Vec<Vec<f64>> = Vec::new();
        for dim in 0..=complex.dimension() as usize {
            if complex.count_dim(dim as isize) > 0 {
                all_evs.push(HodgeLaplacian::compute(&complex, dim).eigenvalues());
            }
        }
        serde_json::to_string(&all_evs).unwrap_or_default()
    }

    /// Compute f-vector and Euler characteristic. Returns JSON.
    #[wasm_bindgen]
    pub fn complex_info(points_flat: &[f64], radius: f64) -> String {
        let points: Vec<[f64; 2]> = points_flat.chunks(2).map(|c| [c[0], c[1]]).collect();
        let complex = SimplicialComplex::vietoris_rips(&points, radius);
        let betti = BettiNumbers::compute(&complex);
        serde_json::to_string(&serde_json::json!({
            "f_vector": complex.f_vector(),
            "euler": complex.euler_characteristic(),
            "betti": betti.values,
            "dimension": complex.dimension(),
            "num_simplices": complex.size(),
        }))
        .unwrap_or_default()
    }
}
