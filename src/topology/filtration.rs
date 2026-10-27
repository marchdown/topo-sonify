use super::simplex::Simplex;
use serde::{Deserialize, Serialize};

/// A simplex with its filtration value — the parameter at which it enters the complex.
///
/// This is the fundamental object in persistent homology. A filtered simplicial
/// complex is a sequence of simplicial complexes K₀ ⊆ K₁ ⊆ ... ⊆ K_n obtained
/// by inserting simplices in filtration order.
///
/// The filtration value of a simplex σ is max{d(v_i, v_j) : v_i, v_j ∈ σ}
/// for the Vietoris-Rips filtration — the diameter of the simplex.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilteredSimplex {
    pub simplex: Simplex,
    pub filtration: f64,
}

impl FilteredSimplex {
    pub fn new(simplex: Simplex, filtration: f64) -> Self {
        Self {
            simplex,
            filtration,
        }
    }

    pub fn dimension(&self) -> isize {
        self.simplex.dimension()
    }
}

/// A filtered simplicial complex: simplices ordered by filtration value,
/// with ties broken by dimension (lower dimension first — a simplex
/// must enter after all its faces).
///
/// This is the input to the standard persistence algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilteredComplex {
    /// Simplices in filtration order. Index = column index in the boundary matrix.
    pub simplices: Vec<FilteredSimplex>,
    /// Map from simplex to its index in the filtration order.
    simplex_to_index: std::collections::HashMap<Vec<usize>, usize>,
}

impl FilteredComplex {
    /// Build a Vietoris-Rips filtration from a point cloud.
    ///
    /// Algorithm:
    /// 1. All vertices enter at filtration value 0.
    /// 2. Each edge (i,j) enters at d(i,j).
    /// 3. Each higher simplex enters at its diameter: max edge weight among its vertices.
    /// 4. Sort by (filtration_value, dimension, lexicographic) to get a valid filtration
    ///    (faces always precede cofaces).
    ///
    /// Complexity: O(n² + S log S) where S = total simplices.
    pub fn vietoris_rips(points: &[[f64; 2]], max_radius: f64) -> Self {
        let n = points.len();
        let mut filtered: Vec<FilteredSimplex> = Vec::new();

        if n == 0 {
            return Self {
                simplices: filtered,
                simplex_to_index: Default::default(),
            };
        }

        // Distance function.
        let dist = |i: usize, j: usize| -> f64 {
            ((points[i][0] - points[j][0]).powi(2) + (points[i][1] - points[j][1]).powi(2)).sqrt()
        };

        // 0-simplices: all vertices at filtration 0.
        for i in 0..n {
            filtered.push(FilteredSimplex::new(Simplex::from_sorted(vec![i]), 0.0));
        }

        // 1-simplices: edges at their distance.
        let mut adj_fwd: Vec<Vec<usize>> = vec![vec![]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dist(i, j);
                if d <= max_radius {
                    filtered.push(FilteredSimplex::new(Simplex::from_sorted(vec![i, j]), d));
                    adj_fwd[i].push(j);
                }
            }
        }

        // 2-simplices: triangles at max edge weight.
        for i in 0..n {
            for idx_j in 0..adj_fwd[i].len() {
                let j = adj_fwd[i][idx_j];
                let dij = dist(i, j);
                for &k in &adj_fwd[i][(idx_j + 1)..] {
                    let djk = dist(j, k);
                    if djk > max_radius {
                        continue;
                    }
                    let dik = dist(i, k);
                    let diameter = dij.max(dik).max(djk);
                    filtered.push(FilteredSimplex::new(
                        Simplex::from_sorted(vec![i, j, k]),
                        diameter,
                    ));
                }
            }
        }

        // 3-simplices: tetrahedra at max edge weight.
        for i in 0..n {
            for idx_j in 0..adj_fwd[i].len() {
                let j = adj_fwd[i][idx_j];
                for pi_k in (idx_j + 1)..adj_fwd[i].len() {
                    let k = adj_fwd[i][pi_k];
                    if dist(j, k) > max_radius {
                        continue;
                    }
                    for &l in &adj_fwd[i][(pi_k + 1)..] {
                        if dist(j, l) <= max_radius && dist(k, l) <= max_radius {
                            let diameter = [
                                dist(i, j),
                                dist(i, k),
                                dist(i, l),
                                dist(j, k),
                                dist(j, l),
                                dist(k, l),
                            ]
                            .into_iter()
                            .fold(0.0_f64, f64::max);
                            filtered.push(FilteredSimplex::new(
                                Simplex::from_sorted(vec![i, j, k, l]),
                                diameter,
                            ));
                        }
                    }
                }
            }
        }

        // Sort: by filtration value, then dimension, then lexicographic.
        // This guarantees faces precede cofaces (lower dim at same filtration).
        filtered.sort_by(|a, b| {
            a.filtration
                .partial_cmp(&b.filtration)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.simplex.dimension().cmp(&b.simplex.dimension()))
                .then_with(|| a.simplex.vertices().cmp(b.simplex.vertices()))
        });

        // Build index map.
        let simplex_to_index: std::collections::HashMap<Vec<usize>, usize> = filtered
            .iter()
            .enumerate()
            .map(|(i, fs)| (fs.simplex.vertices().to_vec(), i))
            .collect();

        Self {
            simplices: filtered,
            simplex_to_index,
        }
    }

    /// Look up the column index of a simplex.
    pub fn index_of(&self, vertices: &[usize]) -> Option<usize> {
        self.simplex_to_index.get(vertices).copied()
    }

    /// Total number of simplices in the filtration.
    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }

    /// The boundary of the j-th simplex as a sorted list of column indices.
    /// Over Z/2Z, the boundary of σ = [v₀,...,vₖ] is Σᵢ [v₀,...,v̂ᵢ,...,vₖ].
    /// We return the indices of the faces in filtration order.
    pub fn boundary_indices(&self, j: usize) -> Vec<usize> {
        let sigma = &self.simplices[j].simplex;
        if sigma.len() <= 1 {
            return vec![]; // 0-simplex has empty boundary
        }
        let mut indices: Vec<usize> = sigma
            .faces()
            .into_iter()
            .filter_map(|(_, face)| self.index_of(face.vertices()))
            .collect();
        indices.sort_unstable();
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtration_order() {
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
        let fc = FilteredComplex::vietoris_rips(&points, 2.0);
        // Vertices first (filt=0), then edges, then triangle.
        assert_eq!(fc.simplices[0].dimension(), 0);
        assert_eq!(fc.simplices[1].dimension(), 0);
        assert_eq!(fc.simplices[2].dimension(), 0);
        // Last simplex should be the triangle (highest filtration or highest dim).
        assert_eq!(fc.simplices.last().unwrap().dimension(), 2);
    }

    #[test]
    fn boundary_indices_correct() {
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
        let fc = FilteredComplex::vietoris_rips(&points, 2.0);
        // Find the triangle's index and check its boundary has 3 entries.
        let tri_idx = fc.simplices.len() - 1;
        let bdry = fc.boundary_indices(tri_idx);
        assert_eq!(bdry.len(), 3); // triangle has 3 edges as boundary
    }
}
