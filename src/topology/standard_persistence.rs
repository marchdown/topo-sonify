use super::filtration::FilteredComplex;
use super::persistence::{PersistenceDiagram, PersistencePair};

/// Standard persistence algorithm (Edelsbrunner-Letscher-Zomorodian 2002,
/// optimized per Zomorodian-Carlsson 2005).
///
/// Given a filtered simplicial complex with m simplices, this computes
/// the complete persistence diagram by reducing the m × m filtration
/// boundary matrix over Z/2Z.
///
/// The algorithm processes columns left to right. Each column j represents
/// the boundary ∂(σⱼ) of the j-th simplex in filtration order. We reduce
/// columns using the "lowest one" pivot: if column j has lowest nonzero
/// entry in row i, and a previous column j' already claimed row i as its
/// pivot, we add column j' to column j (XOR over Z/2Z) to cancel the entry.
///
/// After reduction:
/// - If column j reduces to zero: σⱼ creates a new homology class (birth).
/// - If column j has pivot in row i: σⱼ kills the class born by σᵢ.
///   The persistence pair is (filtration(σᵢ), filtration(σⱼ)).
///
/// Complexity: O(m³) worst case, but typically much faster due to sparsity.
/// The twist optimization (Chen-Kerber 2011) clears columns that are known
/// to reduce to zero, providing significant practical speedup.
pub fn compute_persistence(filtered: &FilteredComplex) -> PersistenceDiagram {
    let m = filtered.len();
    if m == 0 {
        return PersistenceDiagram { pairs: vec![] };
    }

    // Represent each column as a sorted Vec<usize> of nonzero row indices.
    // Sorted representation allows O(n) addition (merge + deduplicate = XOR over Z/2Z).
    let mut columns: Vec<Vec<usize>> = Vec::with_capacity(m);
    for j in 0..m {
        columns.push(filtered.boundary_indices(j));
    }

    // low[j] = the largest row index in column j after reduction, or None if zero.
    // pivot_of_row[i] = j means column j has its pivot (lowest nonzero entry) in row i.
    let mut pivot_of_row: Vec<Option<usize>> = vec![None; m];

    // Track which simplices are "negative" (their column reduces to nonzero)
    // and which are "positive" (their column reduces to zero).
    let mut paired: Vec<bool> = vec![false; m];

    for j in 0..m {
        // Reduce column j.
        loop {
            let low = lowest(&columns[j]);
            match low {
                None => break, // Column is zero → σⱼ is a creator (positive simplex).
                Some(i) => {
                    if let Some(j_prime) = pivot_of_row[i] {
                        // Add column j' to column j (XOR: symmetric difference).
                        let col_prime = columns[j_prime].clone();
                        columns[j] = xor_sorted(&columns[j], &col_prime);
                    } else {
                        // Column j has unique pivot at row i.
                        // This means σⱼ kills the class created by σᵢ.
                        pivot_of_row[i] = Some(j);
                        paired[i] = true;
                        paired[j] = true;
                        break;
                    }
                }
            }
        }
    }

    // Extract persistence pairs.
    let mut pairs: Vec<PersistencePair> = Vec::new();

    for j in 0..m {
        if let Some(i) = lowest(&columns[j]) {
            // Column j has pivot at row i → pair (σᵢ, σⱼ).
            // But only if pivot_of_row[i] == Some(j) (i.e., this is the canonical pivot).
            if pivot_of_row[i] == Some(j) {
                let birth = filtered.simplices[i].filtration;
                let death = filtered.simplices[j].filtration;
                let dim = filtered.simplices[i].dimension() as usize;
                // Skip zero-persistence pairs (birth == death).
                if (death - birth).abs() > 1e-12 {
                    pairs.push(PersistencePair {
                        birth,
                        death,
                        dimension: dim,
                    });
                }
            }
        }
    }

    // Essential classes: unpaired positive simplices (column reduced to zero, not paired).
    for j in 0..m {
        if !paired[j] && columns[j].is_empty() {
            // σⱼ creates a class that never dies. Check it's truly a creator:
            // a simplex is a creator if its boundary reduces to zero.
            // For vertices (dim 0), boundary is empty → always creators.
            // For higher simplices, boundary being zero means a new cycle.
            let birth = filtered.simplices[j].filtration;
            let dim = filtered.simplices[j].dimension() as usize;
            pairs.push(PersistencePair {
                birth,
                death: f64::INFINITY,
                dimension: dim,
            });
        }
    }

    // Sort by persistence (longest-lived first).
    pairs.sort_by(|a, b| {
        b.persistence()
            .partial_cmp(&a.persistence())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    PersistenceDiagram { pairs }
}

/// Lowest nonzero entry in a sorted column (= last element since sorted ascending).
#[inline]
fn lowest(col: &[usize]) -> Option<usize> {
    col.last().copied()
}

/// XOR of two sorted vectors over Z/2Z: symmetric difference.
/// Elements appearing in both cancel (removed); elements in only one survive.
/// Result is sorted.
fn xor_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                // Same element in both → cancels over Z/2Z.
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

/// Compute persistence with the twist optimization (Chen-Kerber 2011).
///
/// Key insight: if simplex σⱼ is paired as a destroyer (negative simplex),
/// then its pivot row i identifies the creator σᵢ. We can then immediately
/// clear column i (set it to zero), because we know σᵢ is paired and its
/// column will never be used as a pivot again.
///
/// This "clearing" step eliminates redundant work and provides 2-10x speedup
/// in practice without changing the output.
pub fn compute_persistence_twist(filtered: &FilteredComplex) -> PersistenceDiagram {
    let m = filtered.len();
    if m == 0 {
        return PersistenceDiagram { pairs: vec![] };
    }

    let mut columns: Vec<Vec<usize>> = Vec::with_capacity(m);
    for j in 0..m {
        columns.push(filtered.boundary_indices(j));
    }

    let mut pivot_of_row: Vec<Option<usize>> = vec![None; m];
    let mut paired: Vec<bool> = vec![false; m];

    // Twist: track which columns have been "cleared" (set to zero because
    // their simplex is known to be paired as a creator).
    let mut cleared: Vec<bool> = vec![false; m];

    for j in 0..m {
        if cleared[j] {
            columns[j].clear();
            continue;
        }

        loop {
            let low = lowest(&columns[j]);
            match low {
                None => break,
                Some(i) => {
                    if let Some(j_prime) = pivot_of_row[i] {
                        let col_prime = columns[j_prime].clone();
                        columns[j] = xor_sorted(&columns[j], &col_prime);
                    } else {
                        pivot_of_row[i] = Some(j);
                        paired[i] = true;
                        paired[j] = true;
                        // TWIST: clear column i since σᵢ is now known to be paired.
                        cleared[i] = true;
                        columns[i].clear();
                        break;
                    }
                }
            }
        }
    }

    // Extract pairs (same as standard algorithm).
    let mut pairs: Vec<PersistencePair> = Vec::new();

    for (j, col) in columns.iter().enumerate() {
        if let Some(&i) = col.last() {
            if pivot_of_row[i] == Some(j) {
                let birth = filtered.simplices[i].filtration;
                let death = filtered.simplices[j].filtration;
                let dim = filtered.simplices[i].dimension() as usize;
                if (death - birth).abs() > 1e-12 {
                    pairs.push(PersistencePair {
                        birth,
                        death,
                        dimension: dim,
                    });
                }
            }
        }
    }

    for j in 0..m {
        if !paired[j] {
            pairs.push(PersistencePair {
                birth: filtered.simplices[j].filtration,
                death: f64::INFINITY,
                dimension: filtered.simplices[j].dimension() as usize,
            });
        }
    }

    pairs.sort_by(|a, b| {
        b.persistence()
            .partial_cmp(&a.persistence())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    PersistenceDiagram { pairs }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_persistence() {
        // Three points forming a triangle.
        // At r=0: 3 components. Edges appear, merging components.
        // Triangle fills the hole.
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
        let fc = FilteredComplex::vietoris_rips(&points, 2.0);
        let pd = compute_persistence(&fc);

        // Should have exactly 1 essential H₀ class (the final component).
        let h0_essential: Vec<_> = pd
            .pairs
            .iter()
            .filter(|p| p.dimension == 0 && p.is_essential())
            .collect();
        assert_eq!(h0_essential.len(), 1);

        // Should have 2 finite H₀ pairs (two components merging).
        let h0_finite: Vec<_> = pd
            .pairs
            .iter()
            .filter(|p| p.dimension == 0 && !p.is_essential())
            .collect();
        assert_eq!(h0_finite.len(), 2);
    }

    #[test]
    fn square_persistence_has_loop() {
        // Four points forming a square — should create a 1-cycle.
        let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let fc = FilteredComplex::vietoris_rips(&points, 2.0);
        let pd = compute_persistence(&fc);

        // Should have at least one H₁ pair (a loop that appears then gets filled).
        let h1: Vec<_> = pd.pairs.iter().filter(|p| p.dimension == 1).collect();
        assert!(
            !h1.is_empty(),
            "Expected at least one H₁ feature for a square"
        );
    }

    #[test]
    fn twist_matches_standard() {
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [1.5, 0.5]];
        let fc = FilteredComplex::vietoris_rips(&points, 3.0);
        let pd_std = compute_persistence(&fc);
        let pd_twist = compute_persistence_twist(&fc);

        // Both should produce the same number of pairs.
        assert_eq!(pd_std.pairs.len(), pd_twist.pairs.len());

        // Same total persistence.
        let tp_std = pd_std.total_persistence();
        let tp_twist = pd_twist.total_persistence();
        assert!((tp_std - tp_twist).abs() < 1e-10);
    }
}
