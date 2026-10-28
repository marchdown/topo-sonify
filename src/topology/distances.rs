use super::persistence::PersistencePair;

/// Bottleneck distance between two persistence diagrams.
///
/// d_B(D₁, D₂) = inf_{γ} sup_{x} ||x - γ(x)||_∞
///
/// where γ ranges over all bijections between D₁ ∪ Δ and D₂ ∪ Δ
/// (each point can be matched to a point on the diagonal Δ).
///
/// The stability theorem guarantees:
///   d_B(Dgm(f), Dgm(g)) ≤ ||f - g||_∞
///
/// Implementation: binary search on the distance threshold δ, checking
/// at each δ whether a perfect matching exists in the bipartite graph
/// where points within L∞ distance δ are connected. The matching check
/// uses the Hopcroft-Karp algorithm.
///
/// Complexity: O(n^{1.5} log n) where n = max(|D₁|, |D₂|).
pub fn bottleneck_distance(d1: &[PersistencePair], d2: &[PersistencePair]) -> f64 {
    // Collect all finite points + diagonal projections.
    let pts1: Vec<[f64; 2]> = d1
        .iter()
        .filter(|p| !p.is_essential())
        .map(|p| [p.birth, p.death])
        .collect();
    let pts2: Vec<[f64; 2]> = d2
        .iter()
        .filter(|p| !p.is_essential())
        .map(|p| [p.birth, p.death])
        .collect();

    if pts1.is_empty() && pts2.is_empty() {
        return 0.0;
    }

    // Collect all candidate distance thresholds.
    // These include: pairwise L∞ distances and distances to the diagonal.
    let mut candidates: Vec<f64> = Vec::new();

    for p in &pts1 {
        candidates.push((p[1] - p[0]) / 2.0); // distance to diagonal
    }
    for p in &pts2 {
        candidates.push((p[1] - p[0]) / 2.0);
    }
    for a in &pts1 {
        for b in &pts2 {
            let d = (a[0] - b[0]).abs().max((a[1] - b[1]).abs());
            candidates.push(d);
        }
    }

    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-14);

    // Binary search for the smallest δ admitting a perfect matching.
    let mut lo = 0;
    let mut hi = candidates.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if has_perfect_matching(&pts1, &pts2, candidates[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    if lo < candidates.len() {
        candidates[lo]
    } else {
        candidates.last().copied().unwrap_or(0.0)
    }
}

/// Check if a perfect matching exists at distance threshold δ.
/// Each point in pts1 can match to a point in pts2 (if L∞ ≤ δ)
/// or to its diagonal projection (if dist_to_diag ≤ δ), and vice versa.
///
/// Models the augmented bipartite graph with n1+n2 nodes on each side:
///   Left:  0..n1 = pts1 real points, n1..n1+n2 = diagonal proxies for pts2
///   Right: 0..n2 = pts2 real points, n2..n2+n1 = diagonal proxies for pts1
///
/// Edges:
///   pts1[i] <-> pts2[j]:      if L∞(pts1[i], pts2[j]) ≤ δ
///   pts1[i] <-> diag_i:       if (death-birth)/2 ≤ δ (right node n2+i)
///   diag_proxy_j <-> pts2[j]: if (death-birth)/2 ≤ δ (left node n1+j to right node j)
///   diag_proxy_j <-> diag_i:  always (0 cost, left n1+j to right n2+i) — unused diagonal pairs
///
/// A perfect matching in this graph means every real point is accounted for.
///
/// Uses Kuhn's augmenting-path algorithm for bipartite matching.
fn has_perfect_matching(pts1: &[[f64; 2]], pts2: &[[f64; 2]], delta: f64) -> bool {
    let n1 = pts1.len();
    let n2 = pts2.len();
    let n = n1 + n2;

    fn augment(
        u: usize,
        adj: &[Vec<usize>],
        match_r: &mut [Option<usize>],
        visited: &mut [bool],
    ) -> bool {
        for &v in &adj[u] {
            if visited[v] {
                continue;
            }
            visited[v] = true;
            if match_r[v].is_none() || augment(match_r[v].unwrap(), adj, match_r, visited) {
                match_r[v] = Some(u);
                return true;
            }
        }
        false
    }

    // Build adjacency list for the augmented bipartite graph.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n1 {
        // pts1[i] can match pts2[j] (right node j)
        for j in 0..n2 {
            let d = (pts1[i][0] - pts2[j][0])
                .abs()
                .max((pts1[i][1] - pts2[j][1]).abs());
            if d <= delta + 1e-12 {
                adj[i].push(j);
            }
        }
        // pts1[i] can go to its own diagonal slot (right node n2+i)
        let diag_dist = (pts1[i][1] - pts1[i][0]).abs() / 2.0;
        if diag_dist <= delta + 1e-12 {
            adj[i].push(n2 + i);
        }
    }

    for j in 0..n2 {
        let left = n1 + j; // diagonal proxy for pts2[j]
                           // pts2[j] goes to diagonal: left node (n1+j) matches right node j
        let diag_dist = (pts2[j][1] - pts2[j][0]).abs() / 2.0;
        if diag_dist <= delta + 1e-12 {
            adj[left].push(j);
        }
        // Diagonal proxy can also match any unused diagonal slot on the right
        // (diagonal-to-diagonal at zero cost).
        for i in 0..n1 {
            adj[left].push(n2 + i);
        }
    }

    // Find maximum matching using augmenting paths.
    let mut match_r: Vec<Option<usize>> = vec![None; n];

    let mut matched = 0;
    for u in 0..n {
        let mut visited = vec![false; n];
        if augment(u, &adj, &mut match_r, &mut visited) {
            matched += 1;
        }
    }

    matched == n
}

/// Wasserstein distance (p-th order) between two persistence diagrams.
///
/// W_p(D₁, D₂) = ( inf_{γ} Σ ||x - γ(x)||_∞^p )^{1/p}
///
/// For p=1: Earth mover's distance.
/// For p=∞: bottleneck distance.
///
/// Uses the Hungarian algorithm for exact computation.
/// Complexity: O(n³).
pub fn wasserstein_distance(d1: &[PersistencePair], d2: &[PersistencePair], p: f64) -> f64 {
    let pts1: Vec<[f64; 2]> = d1
        .iter()
        .filter(|p| !p.is_essential())
        .map(|p| [p.birth, p.death])
        .collect();
    let pts2: Vec<[f64; 2]> = d2
        .iter()
        .filter(|p| !p.is_essential())
        .map(|p| [p.birth, p.death])
        .collect();

    let n1 = pts1.len();
    let n2 = pts2.len();
    let n = n1.max(n2);

    if n == 0 {
        return 0.0;
    }

    // Build cost matrix: n×n for a balanced assignment.
    // Pad the smaller set with diagonal points.
    // Cost of matching pt to diagonal = (death - birth) / 2 (L∞ to projection).
    let mut cost = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            let c = if i < n1 && j < n2 {
                // Match pts1[i] to pts2[j].
                let d = (pts1[i][0] - pts2[j][0])
                    .abs()
                    .max((pts1[i][1] - pts2[j][1]).abs());
                d.powf(p)
            } else if i < n1 {
                // Match pts1[i] to diagonal.
                ((pts1[i][1] - pts1[i][0]) / 2.0).powf(p)
            } else if j < n2 {
                // Match pts2[j] to diagonal.
                ((pts2[j][1] - pts2[j][0]) / 2.0).powf(p)
            } else {
                // Both are padding (diagonal to diagonal) → 0 cost.
                0.0
            };
            cost[i][j] = c;
        }
    }

    // Hungarian algorithm (Kuhn-Munkres) for minimum-weight perfect matching.
    let total_cost = hungarian(&cost);
    total_cost.powf(1.0 / p)
}

/// Hungarian algorithm for minimum-weight perfect matching on an n×n cost matrix.
/// Returns the total cost of the optimal assignment.
fn hungarian(cost: &[Vec<f64>]) -> f64 {
    let n = cost.len();
    if n == 0 {
        return 0.0;
    }

    let inf = f64::MAX / 2.0;
    // u[i], v[j]: potentials for row i and column j.
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    // p[j]: row assigned to column j (1-indexed, 0 = unassigned).
    let mut p = vec![0usize; n + 1];
    // way[j]: column that was used to reach column j in the augmenting path.
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut min_v = vec![inf; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;

            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if cur < min_v[j] {
                    min_v[j] = cur;
                    way[j] = j0;
                }
                if min_v[j] < delta {
                    delta = min_v[j];
                    j1 = j;
                }
            }

            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_v[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut result = 0.0;
    for j in 1..=n {
        if p[j] != 0 {
            result += cost[p[j] - 1][j - 1];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_diagrams_zero_distance() {
        let pairs = vec![
            PersistencePair {
                birth: 0.0,
                death: 1.0,
                dimension: 0,
            },
            PersistencePair {
                birth: 0.5,
                death: 2.0,
                dimension: 1,
            },
        ];
        assert!((bottleneck_distance(&pairs, &pairs) - 0.0).abs() < 1e-10);
        assert!((wasserstein_distance(&pairs, &pairs, 1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn wasserstein_simple() {
        let d1 = vec![PersistencePair {
            birth: 0.0,
            death: 1.0,
            dimension: 0,
        }];
        let d2 = vec![PersistencePair {
            birth: 0.0,
            death: 2.0,
            dimension: 0,
        }];
        // The only finite pair differs by 1 in the death coordinate.
        let w1 = wasserstein_distance(&d1, &d2, 1.0);
        assert!((w1 - 1.0).abs() < 1e-8);
    }

    #[test]
    fn bottleneck_shifted() {
        let d1 = vec![PersistencePair {
            birth: 0.0,
            death: 1.0,
            dimension: 0,
        }];
        let d2 = vec![PersistencePair {
            birth: 0.0,
            death: 1.5,
            dimension: 0,
        }];
        let bn = bottleneck_distance(&d1, &d2);
        assert!((bn - 0.5).abs() < 1e-8);
    }
}
