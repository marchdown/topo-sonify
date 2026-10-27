use super::complex::SimplicialComplex;

/// The k-th Hodge Laplacian on a simplicial complex.
///
/// L_k = ∂_{k+1} ∂_{k+1}ᵀ + ∂_kᵀ ∂_k
///
/// where ∂_k is the k-th boundary operator. The two terms are:
/// - L_k^up   = ∂_{k+1} ∂_{k+1}ᵀ  (upper Laplacian, detects k-dimensional "curl")
/// - L_k^down = ∂_kᵀ ∂_k            (lower Laplacian, detects k-dimensional "gradient")
///
/// Key properties (discrete Hodge theorem):
/// - ker(L_k) ≅ H_k(K; ℝ)  — the harmonic k-chains are exactly the homology.
/// - dim(ker(L_k)) = β_k     — the k-th Betti number.
/// - The nonzero eigenvalues encode geometric information beyond topology.
///
/// For k=0, L_0 = ∂₁ ∂₁ᵀ is the graph Laplacian. Its eigenvalues determine:
/// - λ₁ (algebraic connectivity / Fiedler value): how well-connected the graph is
/// - The spectral gap λ₁ - λ₀: related to mixing time, Cheeger constant
/// - The full spectrum: determines heat diffusion, random walks, spectral clustering
///
/// Over ℝ (not Z/2Z) for spectral correctness. Signs matter here.
pub struct HodgeLaplacian {
    /// The dense matrix L_k, stored row-major.
    pub matrix: Vec<Vec<f64>>,
    /// Dimension of the matrix (number of k-simplices).
    pub size: usize,
    /// The homological dimension k.
    pub dimension: usize,
}

impl HodgeLaplacian {
    /// Compute the k-th Hodge Laplacian of a simplicial complex.
    ///
    /// L_k = ∂_{k+1} ∂_{k+1}ᵀ + ∂_kᵀ ∂_k
    ///
    /// We compute the signed boundary matrices over ℝ (not Z/2Z).
    pub fn compute(complex: &SimplicialComplex, k: usize) -> Self {
        let k_simplices = complex.simplices_of_dim(k as isize);
        let n = k_simplices.len();

        let mut matrix = vec![vec![0.0; n]; n];

        // L_k^down = ∂_kᵀ ∂_k (from faces below)
        if k > 0 {
            let bdry_k = signed_boundary_matrix(complex, k);
            // ∂_kᵀ ∂_k = Bᵀ B
            let rows_b = bdry_k.len();
            let cols_b = if rows_b > 0 { bdry_k[0].len() } else { 0 };
            for i in 0..cols_b {
                for j in 0..cols_b {
                    let mut sum = 0.0;
                    for r in 0..rows_b {
                        sum += bdry_k[r][i] * bdry_k[r][j];
                    }
                    matrix[i][j] += sum;
                }
            }
        }

        // L_k^up = ∂_{k+1} ∂_{k+1}ᵀ (from cofaces above)
        let d = complex.dimension();
        if (k as isize) < d {
            let bdry_k1 = signed_boundary_matrix(complex, k + 1);
            // ∂_{k+1} ∂_{k+1}ᵀ = B B^T where B = ∂_{k+1}
            let rows_b = bdry_k1.len();
            let cols_b = if rows_b > 0 { bdry_k1[0].len() } else { 0 };
            for i in 0..rows_b {
                for j in 0..rows_b {
                    let mut sum = 0.0;
                    for c in 0..cols_b {
                        sum += bdry_k1[i][c] * bdry_k1[j][c];
                    }
                    matrix[i][j] += sum;
                }
            }
        }

        Self {
            matrix,
            size: n,
            dimension: k,
        }
    }

    /// Compute all eigenvalues via the Jacobi eigenvalue algorithm.
    /// Returns eigenvalues sorted in ascending order.
    ///
    /// The Jacobi algorithm is ideal for small symmetric matrices (which L_k always is).
    /// It iteratively applies Givens rotations to zero out off-diagonal elements,
    /// converging to diagonal form whose entries are the eigenvalues.
    pub fn eigenvalues(&self) -> Vec<f64> {
        if self.size == 0 {
            return vec![];
        }
        if self.size == 1 {
            return vec![self.matrix[0][0]];
        }

        let n = self.size;
        let mut a = self.matrix.clone();

        // Jacobi iteration: repeatedly zero out the largest off-diagonal element.
        let max_iters = 100 * n * n;
        for _ in 0..max_iters {
            // Find the largest off-diagonal element.
            let mut max_val = 0.0_f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            // Convergence check.
            if max_val < 1e-14 {
                break;
            }

            // Compute Givens rotation angle to zero out a[p][q].
            let (c, s) = if (a[p][p] - a[q][q]).abs() < 1e-15 {
                // theta = pi/4
                let c = std::f64::consts::FRAC_1_SQRT_2;
                (c, c)
            } else {
                let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                // t = sign(tau) / (|tau| + sqrt(1 + tau^2))
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                (c, s)
            };

            // Apply Jacobi rotation: A' = Jᵀ A J
            // Update rows/columns p and q.
            for r in 0..n {
                if r == p || r == q {
                    continue;
                }
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
            let app = a[p][p];
            let aqq = a[q][q];
            let apq = a[p][q];
            a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
        }

        // Eigenvalues are the diagonal elements.
        let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();

        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Clamp near-zero eigenvalues.
        for ev in &mut eigenvalues {
            if ev.abs() < 1e-10 {
                *ev = 0.0;
            }
        }
        eigenvalues
    }

    /// Number of zero eigenvalues = β_k (discrete Hodge theorem).
    pub fn kernel_dimension(&self) -> usize {
        self.eigenvalues()
            .iter()
            .filter(|&&ev| ev.abs() < 1e-10)
            .count()
    }

    /// The spectral gap: λ₁ - λ₀. For the graph Laplacian (k=0),
    /// this is the algebraic connectivity (Fiedler value).
    pub fn spectral_gap(&self) -> f64 {
        let evs = self.eigenvalues();
        if evs.len() < 2 {
            return 0.0;
        }
        evs[1] - evs[0]
    }
}

/// Compute the signed boundary matrix ∂_k over ℝ (not Z/2Z).
///
/// ∂_k(σ) = Σᵢ (-1)ⁱ dᵢ(σ), where dᵢ removes the i-th vertex.
/// The sign (-1)ⁱ is crucial for the Hodge Laplacian — it encodes orientation.
fn signed_boundary_matrix(complex: &SimplicialComplex, k: usize) -> Vec<Vec<f64>> {
    let k_simplices = complex.simplices_of_dim(k as isize);
    let km1_simplices = complex.simplices_of_dim(k as isize - 1);

    if k_simplices.is_empty() || km1_simplices.is_empty() {
        return vec![];
    }

    // Row index map for (k-1)-simplices.
    let row_index: std::collections::HashMap<&[usize], usize> = km1_simplices
        .iter()
        .enumerate()
        .map(|(i, s)| (s.vertices(), i))
        .collect();

    let rows = km1_simplices.len();
    let cols = k_simplices.len();
    let mut matrix = vec![vec![0.0; cols]; rows];

    for (j, sigma) in k_simplices.iter().enumerate() {
        for (i, face) in sigma.faces() {
            if let Some(&row) = row_index.get(face.vertices()) {
                // Sign: (-1)^i where i is the position of the removed vertex.
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                matrix[row][j] = sign;
            }
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_laplacian_triangle() {
        // Complete graph K₃ (filled triangle) — L₀ should have eigenvalues 0, 3, 3.
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1, 2]);
        let l0 = HodgeLaplacian::compute(&k, 0);
        let evs = l0.eigenvalues();
        assert_eq!(evs.len(), 3);
        assert!((evs[0] - 0.0).abs() < 1e-8); // one zero eigenvalue (connected)
        assert!((evs[1] - 3.0).abs() < 0.1);
        assert!((evs[2] - 3.0).abs() < 0.1);
    }

    #[test]
    fn hodge_betti_hollow_tetrahedron() {
        // S² has β₀=1, β₁=0, β₂=1.
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1, 2]);
        k.insert_simplex(&[0, 1, 3]);
        k.insert_simplex(&[0, 2, 3]);
        k.insert_simplex(&[1, 2, 3]);

        assert_eq!(HodgeLaplacian::compute(&k, 0).kernel_dimension(), 1);
        assert_eq!(HodgeLaplacian::compute(&k, 1).kernel_dimension(), 0);
        assert_eq!(HodgeLaplacian::compute(&k, 2).kernel_dimension(), 1);
    }

    #[test]
    fn spectral_gap_connected() {
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1]);
        k.insert_simplex(&[1, 2]);
        k.insert_simplex(&[0, 2]);
        let l0 = HodgeLaplacian::compute(&k, 0);
        assert!(l0.spectral_gap() > 0.0); // connected → positive spectral gap
    }

    #[test]
    fn spectral_gap_disconnected() {
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1]);
        k.insert_simplex(&[2, 3]);
        let l0 = HodgeLaplacian::compute(&k, 0);
        assert!((l0.spectral_gap() - 0.0).abs() < 1e-8); // disconnected → zero gap
    }
}
