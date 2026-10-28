use super::complex::SimplicialComplex;
use serde::{Deserialize, Serialize};

/// Betti numbers β_k = rank(H_k) — the fundamental topological invariants
/// that drive our sonification.
///
/// β₀ = number of connected components
/// β₁ = number of independent loops / tunnels
/// β₂ = number of enclosed voids / cavities
/// β_k = number of k-dimensional "holes"
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BettiNumbers {
    pub values: Vec<usize>,
}

impl BettiNumbers {
    /// Compute Betti numbers β₀, β₁, ..., β_d from a simplicial complex.
    ///
    /// Uses two optimizations:
    /// 1. β₀ is computed via union-find in O(n·α(n)) instead of O(n³) matrix reduction.
    /// 2. For k ≥ 1, boundary matrix ranks are precomputed once and reused:
    ///    β_k = (dim C_k - rank ∂_k) - rank ∂_{k+1}
    pub fn compute(complex: &SimplicialComplex) -> Self {
        let d = complex.dimension();
        if d < 0 {
            return Self { values: vec![] };
        }

        let mut values = Vec::with_capacity((d + 1) as usize);

        // β₀ via union-find — O(V·α(V)) instead of building ∂₁ and reducing.
        values.push(complex.connected_components());

        if d >= 1 {
            // Precompute rank(∂_k) for k = 1, ..., d.
            let mut ranks: Vec<usize> = Vec::with_capacity(d as usize);
            for k in 1..=d {
                ranks.push(complex.boundary_matrix_z2(k).rank_z2());
            }
            // ranks[0] = rank(∂_1), ranks[1] = rank(∂_2), ...

            for k in 1..=d as usize {
                let dim_ck = complex.count_dim(k as isize);
                let rank_dk = ranks[k - 1]; // rank(∂_k)
                let rank_dk1 = if k < d as usize { ranks[k] } else { 0 }; // rank(∂_{k+1})
                let betti = dim_ck.saturating_sub(rank_dk).saturating_sub(rank_dk1);
                values.push(betti);
            }
        }

        Self { values }
    }

    /// β₀: connected components
    pub fn b0(&self) -> usize {
        self.values.first().copied().unwrap_or(0)
    }

    /// β₁: loops
    pub fn b1(&self) -> usize {
        self.values.get(1).copied().unwrap_or(0)
    }

    /// β₂: voids
    pub fn b2(&self) -> usize {
        self.values.get(2).copied().unwrap_or(0)
    }

    /// Total Betti number Σ β_k — a measure of total topological complexity.
    pub fn total(&self) -> usize {
        self.values.iter().sum()
    }

    /// Euler characteristic from Betti numbers: χ = Σ (-1)^k β_k
    pub fn euler_characteristic(&self) -> isize {
        self.values
            .iter()
            .enumerate()
            .map(|(k, &b)| {
                if k % 2 == 0 {
                    b as isize
                } else {
                    -(b as isize)
                }
            })
            .sum()
    }
}

/// A chain complex C_n → C_{n-1} → ... → C_0 with boundary maps.
#[derive(Debug, Clone)]
pub struct ChainComplex {
    pub dimensions: Vec<usize>,     // dim(C_k)
    pub boundary_ranks: Vec<usize>, // rank(∂_k) for k = 1, ..., n
}

impl ChainComplex {
    pub fn from_complex(complex: &SimplicialComplex) -> Self {
        let d = complex.dimension();
        let dimensions: Vec<usize> = (0..=d).map(|k| complex.simplices_of_dim(k).len()).collect();
        let boundary_ranks: Vec<usize> = (1..=d)
            .map(|k| complex.boundary_matrix_z2(k).rank_z2())
            .collect();

        Self {
            dimensions,
            boundary_ranks,
        }
    }
}

/// Homology group metadata for sonification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomologyGroup {
    pub dimension: usize,
    pub rank: usize,
    /// Representative cycles as lists of simplex vertex sets.
    pub generators: Vec<Vec<Vec<usize>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_betti() {
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1, 2]); // filled triangle
        let b = BettiNumbers::compute(&k);
        assert_eq!(b.b0(), 1); // one component
        assert_eq!(b.b1(), 0); // no loops (it's filled)
    }

    #[test]
    fn triangle_boundary_betti() {
        // Just the boundary of a triangle — a cycle!
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1]);
        k.insert_simplex(&[1, 2]);
        k.insert_simplex(&[0, 2]);
        let b = BettiNumbers::compute(&k);
        assert_eq!(b.b0(), 1); // one component
        assert_eq!(b.b1(), 1); // one loop
    }

    #[test]
    fn two_components() {
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1]);
        k.insert_simplex(&[2, 3]);
        let b = BettiNumbers::compute(&k);
        assert_eq!(b.b0(), 2);
        assert_eq!(b.b1(), 0);
    }

    #[test]
    fn hollow_tetrahedron_betti() {
        // S² → β₀=1, β₁=0, β₂=1
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1, 2]);
        k.insert_simplex(&[0, 1, 3]);
        k.insert_simplex(&[0, 2, 3]);
        k.insert_simplex(&[1, 2, 3]);
        let b = BettiNumbers::compute(&k);
        assert_eq!(b.b0(), 1);
        assert_eq!(b.b1(), 0);
        assert_eq!(b.b2(), 1); // one void!
        assert_eq!(b.euler_characteristic(), 2); // χ(S²) = 2
    }
}
