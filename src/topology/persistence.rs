use super::complex::SimplicialComplex;
use super::homology::BettiNumbers;
use serde::{Deserialize, Serialize};

/// A filtration value: the parameter at which a simplex enters the complex.
pub type FiltrationValue = f64;

/// A persistence pair (birth, death) representing a topological feature
/// that appears at `birth` and disappears at `death`.
/// If death = f64::INFINITY, the feature persists forever (essential class).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersistencePair {
    pub birth: FiltrationValue,
    pub death: FiltrationValue,
    pub dimension: usize,
}

impl PersistencePair {
    /// The lifetime of the feature.
    pub fn persistence(&self) -> f64 {
        if self.death.is_infinite() {
            f64::INFINITY
        } else {
            self.death - self.birth
        }
    }

    /// Midpoint of the interval (for finite pairs).
    pub fn midpoint(&self) -> f64 {
        if self.death.is_infinite() {
            self.birth
        } else {
            (self.birth + self.death) / 2.0
        }
    }

    /// Is this an essential class (never dies)?
    pub fn is_essential(&self) -> bool {
        self.death.is_infinite()
    }
}

/// A persistence diagram: the multiset of all persistence pairs.
/// This is the central data structure for topology-driven sonification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
}

impl PersistenceDiagram {
    /// Compute a persistence diagram from a Vietoris-Rips filtration on a point cloud.
    ///
    /// Algorithm overview:
    /// 1. Compute all pairwise distances and sort edges by weight.
    /// 2. Use the sorted edge weights as filtration values (no uniform sampling —
    ///    topology only changes at edge insertions, so we sample exactly the
    ///    critical radii, eliminating discretization error).
    /// 3. At each critical radius, incrementally build the VR complex and
    ///    recompute Betti numbers. Track births and deaths of features.
    ///
    /// Complexity: O(E · T(homology)) where E = number of distinct edge weights
    /// and T(homology) is the cost of a Betti number computation. This is a
    /// significant improvement over uniform sampling, which misses critical radii
    /// and wastes computation on non-critical steps.
    ///
    /// Note: a proper implementation would use the standard persistence algorithm
    /// (column reduction on the full filtration boundary matrix) rather than
    /// recomputing Betti numbers at each step. See docs/RESEARCH-HANDOFF.md for
    /// the plan to implement Ripser-style persistence.
    pub fn from_point_cloud(points: &[[f64; 2]], _num_steps: usize) -> Self {
        let n = points.len();
        if n == 0 {
            return Self { pairs: vec![] };
        }

        // Step 1: Compute all pairwise distances.
        let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = ((points[i][0] - points[j][0]).powi(2)
                    + (points[i][1] - points[j][1]).powi(2))
                .sqrt();
                edges.push((d, i, j));
            }
        }

        // Step 2: Sort edges by distance. These are the critical filtration values.
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Deduplicate consecutive equal distances (they form a single filtration step).
        let mut critical_radii: Vec<f64> = Vec::new();
        critical_radii.push(0.0); // r=0: all vertices, no edges
        for &(d, _, _) in &edges {
            if critical_radii
                .last()
                .is_none_or(|&last| (d - last).abs() > 1e-12)
            {
                critical_radii.push(d);
            }
        }

        // Step 3: Build VR complex at each critical radius and track Betti changes.
        let mut prev_betti = BettiNumbers { values: vec![] };
        let mut active_features: Vec<Vec<f64>> = Vec::new(); // per dim: birth times
        let mut pairs = Vec::new();

        for &radius in &critical_radii {
            let complex = SimplicialComplex::vietoris_rips(points, radius);
            let betti = BettiNumbers::compute(&complex);

            // Grow active_features to match dimensionality
            while active_features.len() < betti.values.len() {
                active_features.push(Vec::new());
            }

            for (dim, active) in active_features
                .iter_mut()
                .enumerate()
                .take(betti.values.len())
            {
                let prev = if dim < prev_betti.values.len() {
                    prev_betti.values[dim]
                } else {
                    0
                };
                let curr = betti.values[dim];

                if curr > prev {
                    for _ in 0..(curr - prev) {
                        active.push(radius);
                    }
                } else if curr < prev {
                    // Death: elder rule — kill youngest (most recently born) first
                    for _ in 0..(prev - curr) {
                        if let Some(birth) = active.pop() {
                            // Skip trivial pairs with zero persistence
                            if (radius - birth).abs() > 1e-12 {
                                pairs.push(PersistencePair {
                                    birth,
                                    death: radius,
                                    dimension: dim,
                                });
                            }
                        }
                    }
                }
            }

            prev_betti = betti;
        }

        // Remaining features are essential (never die)
        for (dim, features) in active_features.iter().enumerate() {
            for &birth in features {
                pairs.push(PersistencePair {
                    birth,
                    death: f64::INFINITY,
                    dimension: dim,
                });
            }
        }

        // Sort by persistence (longest-lived first)
        pairs.sort_by(|a, b| {
            b.persistence()
                .partial_cmp(&a.persistence())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Self { pairs }
    }

    /// Filter pairs by dimension.
    pub fn pairs_in_dim(&self, dim: usize) -> Vec<&PersistencePair> {
        self.pairs.iter().filter(|p| p.dimension == dim).collect()
    }

    /// Total persistence: sum of all finite lifetimes.
    pub fn total_persistence(&self) -> f64 {
        self.pairs
            .iter()
            .filter(|p| !p.is_essential())
            .map(|p| p.persistence())
            .sum()
    }

    /// Maximum persistence (longest-lived finite feature).
    pub fn max_persistence(&self) -> f64 {
        self.pairs
            .iter()
            .filter(|p| !p.is_essential())
            .map(|p| p.persistence())
            .fold(0.0_f64, f64::max)
    }

    /// Persistence landscape value at a given point.
    /// λ_k(t) for the k-th landscape function at parameter t.
    pub fn landscape(&self, dim: usize, k: usize, t: f64) -> f64 {
        let mut values: Vec<f64> = self
            .pairs_in_dim(dim)
            .iter()
            .filter(|p| !p.is_essential())
            .map(|p| {
                let mid = p.midpoint();
                let half_life = p.persistence() / 2.0;
                if (t - mid).abs() < half_life {
                    half_life - (t - mid).abs()
                } else {
                    0.0
                }
            })
            .collect();
        values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        values.get(k).copied().unwrap_or(0.0)
    }

    /// Persistence entropy: H = -Σ p_i log(p_i) where p_i = pers_i / total_pers
    pub fn persistence_entropy(&self) -> f64 {
        let total = self.total_persistence();
        if total == 0.0 {
            return 0.0;
        }
        self.pairs
            .iter()
            .filter(|p| !p.is_essential())
            .map(|p| {
                let prob = p.persistence() / total;
                if prob > 0.0 {
                    -prob * prob.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistence_pair_basics() {
        let p = PersistencePair {
            birth: 0.5,
            death: 1.5,
            dimension: 1,
        };
        assert_eq!(p.persistence(), 1.0);
        assert_eq!(p.midpoint(), 1.0);
        assert!(!p.is_essential());
    }

    #[test]
    fn essential_pair() {
        let p = PersistencePair {
            birth: 0.0,
            death: f64::INFINITY,
            dimension: 0,
        };
        assert!(p.is_essential());
        assert_eq!(p.persistence(), f64::INFINITY);
    }

    #[test]
    fn point_cloud_persistence() {
        // Square with points — should detect a loop
        let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let pd = PersistenceDiagram::from_point_cloud(&points, 20);
        // Should have some β₀ features (components merging) and potentially a β₁ feature (loop)
        assert!(!pd.pairs.is_empty());
        let h0_pairs = pd.pairs_in_dim(0);
        assert!(!h0_pairs.is_empty());
    }
}
