use super::complex::SimplicialComplex;
use super::simplex::Simplex;

/// A discrete Morse function on a simplicial complex (Forman 1998).
///
/// A discrete Morse function f: K → ℝ assigns a real value to each simplex
/// such that for each k-simplex σ:
///   - At most one (k+1)-coface τ > σ satisfies f(τ) ≤ f(σ)
///   - At most one (k-1)-face ρ < σ satisfies f(ρ) ≥ f(σ)
///
/// A simplex σ is **critical** if it has NO coface τ > σ with f(τ) ≤ f(σ)
/// AND no face ρ < σ with f(ρ) ≥ f(σ).
///
/// The Morse inequalities guarantee: c_k ≥ β_k, where c_k is the number
/// of critical k-simplices. An optimal discrete Morse function minimizes c_k.
///
/// A **discrete gradient vector field** V is the collection of pairs (σ, τ)
/// where σ is a k-simplex, τ is a (k+1)-coface, and they are matched
/// (f(τ) ≤ f(σ)). Unpaired simplices are critical.
pub struct DiscreteMorseFunction {
    /// The function value f(σ) for each simplex, indexed by a unique simplex ID.
    pub values: Vec<(Simplex, f64)>,
    /// The gradient vector field: pairs (σ, τ) where dim(τ) = dim(σ) + 1.
    pub gradient_pairs: Vec<(Simplex, Simplex)>,
    /// Critical simplices (unpaired).
    pub critical: Vec<Simplex>,
}

impl DiscreteMorseFunction {
    /// Construct a discrete Morse function from the lower-star filtration
    /// of a function on the vertices.
    ///
    /// Given f: V → ℝ on vertices, extend to all simplices by:
    ///   f(σ) = max{f(v) : v ∈ σ}  (upper-star / sublevel extension)
    ///
    /// This always gives a valid discrete Morse function when combined
    /// with a lexicographic tiebreaker on vertices.
    ///
    /// The gradient pairs are constructed greedily: process simplices in
    /// order of increasing f-value. Each unpaired face σ is paired with
    /// its lowest unpaired coface τ if f(τ) ≤ f(σ) + ε (within the same
    /// sublevel set).
    pub fn from_vertex_function(complex: &SimplicialComplex, vertex_values: &[f64]) -> Self {
        let _d = complex.dimension();
        let mut all_simplices: Vec<(Simplex, f64)> = Vec::new();

        // Assign values to all simplices: f(σ) = max{f(v) : v ∈ σ}.
        for s in complex.iter() {
            let val = s
                .vertex_iter()
                .filter_map(|v| vertex_values.get(v))
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            all_simplices.push((s.clone(), val));
        }

        // Sort by (value, dimension, lexicographic) — lower dimension first
        // at equal values so faces precede cofaces.
        all_simplices.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.dimension().cmp(&b.0.dimension()))
                .then_with(|| a.0.vertices().cmp(b.0.vertices()))
        });

        // Pairing via the "unique unpaired face" rule (standard discrete Morse
        // construction): process simplices in order of increasing f-value.
        // A simplex τ is paired with a face σ if σ is the *unique* unpaired
        // face of τ at the time τ is processed. This guarantees a valid
        // discrete Morse function satisfying the Morse inequalities.
        let mut paired: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();
        let mut gradient_pairs: Vec<(Simplex, Simplex)> = Vec::new();
        let mut critical: Vec<Simplex> = Vec::new();

        // Process simplices in filtration order (increasing value, lower dim first).
        for (sigma, _val) in &all_simplices {
            let key = sigma.vertices().to_vec();
            if paired.contains(&key) {
                continue;
            }

            if sigma.dimension() > 0 {
                // Count unpaired faces of sigma.
                let faces: Vec<Simplex> = sigma.faces().into_iter().map(|(_, f)| f).collect();
                let unpaired_faces: Vec<&Simplex> = faces
                    .iter()
                    .filter(|f| !paired.contains(f.vertices()))
                    .collect();

                if unpaired_faces.len() == 1 {
                    // Exactly one unpaired face: pair (face, sigma).
                    let face = unpaired_faces[0];
                    paired.insert(face.vertices().to_vec());
                    paired.insert(key);
                    gradient_pairs.push((face.clone(), sigma.clone()));
                    continue;
                }
            }

            // sigma is critical (either a vertex with no pairing opportunity,
            // or a higher simplex with 0 or ≥2 unpaired faces).
            paired.insert(key);
            critical.push(sigma.clone());
        }

        Self {
            values: all_simplices,
            gradient_pairs,
            critical,
        }
    }

    /// Count critical simplices by dimension.
    pub fn critical_counts(&self) -> Vec<usize> {
        let max_dim = self
            .critical
            .iter()
            .map(|s| s.dimension())
            .max()
            .unwrap_or(-1);
        if max_dim < 0 {
            return vec![];
        }
        let mut counts = vec![0usize; max_dim as usize + 1];
        for s in &self.critical {
            counts[s.dimension() as usize] += 1;
        }
        counts
    }

    /// Verify the weak Morse inequalities: c_k ≥ β_k for all k.
    pub fn verify_morse_inequalities(&self, betti: &[usize]) -> bool {
        let counts = self.critical_counts();
        for (k, &b) in betti.iter().enumerate() {
            let c = counts.get(k).copied().unwrap_or(0);
            if c < b {
                return false;
            }
        }
        true
    }

    /// Verify the strong Morse inequality (Euler form):
    /// Σ (-1)^k c_k = χ = Σ (-1)^k β_k.
    pub fn euler_from_critical(&self) -> isize {
        self.critical_counts()
            .iter()
            .enumerate()
            .map(|(k, &c)| {
                if k % 2 == 0 {
                    c as isize
                } else {
                    -(c as isize)
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::homology::BettiNumbers;

    #[test]
    fn morse_triangle() {
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1, 2]);
        let morse = DiscreteMorseFunction::from_vertex_function(&k, &[0.0, 1.0, 2.0]);

        // Weak Morse inequalities.
        let betti = BettiNumbers::compute(&k);
        assert!(morse.verify_morse_inequalities(&betti.values));

        // Strong Morse inequality (Euler).
        assert_eq!(morse.euler_from_critical(), k.euler_characteristic());
    }

    #[test]
    fn morse_hollow_tetrahedron() {
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1, 2]);
        k.insert_simplex(&[0, 1, 3]);
        k.insert_simplex(&[0, 2, 3]);
        k.insert_simplex(&[1, 2, 3]);
        let morse = DiscreteMorseFunction::from_vertex_function(&k, &[0.0, 1.0, 2.0, 3.0]);

        let betti = BettiNumbers::compute(&k);
        assert!(morse.verify_morse_inequalities(&betti.values));
        assert_eq!(morse.euler_from_critical(), 2); // χ(S²)
    }
}
