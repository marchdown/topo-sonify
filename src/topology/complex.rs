use super::simplex::Simplex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

/// A simplicial complex: a collection of simplices closed under taking faces.
///
/// Maintains a dimension-indexed cache alongside the canonical set.
/// The cache stores indices into a flat simplex vector rather than
/// cloning simplices, avoiding double memory usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplicialComplex {
    /// Canonical store — flat vector, insertion order.
    all: Vec<Simplex>,
    /// Dedup set for O(log n) membership checks.
    membership: BTreeSet<Simplex>,
    /// Indices into `all`, grouped by dimension: by_dim[k] = indices of k-simplices.
    by_dim: Vec<Vec<usize>>,
    max_dim: isize,
}

impl SimplicialComplex {
    pub fn new() -> Self {
        Self {
            all: Vec::new(),
            membership: BTreeSet::new(),
            by_dim: Vec::new(),
            max_dim: -1,
        }
    }

    /// Insert a simplex and all its faces (closure property).
    /// Iterative worklist. Each simplex is inserted at most once.
    pub fn insert(&mut self, simplex: Simplex) {
        let mut worklist = vec![simplex];
        while let Some(s) = worklist.pop() {
            if self.membership.contains(&s) {
                continue;
            }
            // Push faces onto worklist before inserting s.
            if s.len() > 1 {
                for (_, face) in s.faces() {
                    if !self.membership.contains(&face) {
                        worklist.push(face);
                    }
                }
            }
            let dim = s.dimension();
            if dim > self.max_dim {
                self.max_dim = dim;
            }
            // Grow by_dim if needed.
            while self.by_dim.len() <= dim as usize {
                self.by_dim.push(Vec::new());
            }
            let idx = self.all.len();
            self.by_dim[dim as usize].push(idx);
            self.membership.insert(s.clone());
            self.all.push(s);
        }
    }

    /// Insert a simplex from raw vertices.
    #[inline]
    pub fn insert_simplex(&mut self, vertices: &[usize]) {
        self.insert(Simplex::new(vertices.iter().copied()));
    }

    /// All simplices of a given dimension — returns references into the flat store.
    pub fn simplices_of_dim(&self, dim: isize) -> Vec<&Simplex> {
        if dim < 0 || dim as usize >= self.by_dim.len() {
            return Vec::new();
        }
        self.by_dim[dim as usize]
            .iter()
            .map(|&i| &self.all[i])
            .collect()
    }

    /// Number of k-simplices for a given dimension.
    #[inline]
    pub fn count_dim(&self, dim: isize) -> usize {
        if dim < 0 || dim as usize >= self.by_dim.len() {
            0
        } else {
            self.by_dim[dim as usize].len()
        }
    }

    /// The f-vector (f₀, f₁, ..., f_d). Single indexed lookup per dimension.
    pub fn f_vector(&self) -> Vec<usize> {
        (0..=self.max_dim).map(|d| self.count_dim(d)).collect()
    }

    /// Euler characteristic: χ = Σ (-1)^k f_k. Single pass.
    pub fn euler_characteristic(&self) -> isize {
        let mut chi: isize = 0;
        for (k, counts) in self.by_dim.iter().enumerate() {
            if k % 2 == 0 {
                chi += counts.len() as isize;
            } else {
                chi -= counts.len() as isize;
            }
        }
        chi
    }

    /// Maximum dimension of any simplex in the complex.
    #[inline]
    pub fn dimension(&self) -> isize {
        self.max_dim
    }

    /// Total number of simplices.
    #[inline]
    pub fn size(&self) -> usize {
        self.all.len()
    }

    /// All vertices (0-simplices).
    pub fn vertices(&self) -> Vec<usize> {
        self.simplices_of_dim(0)
            .iter()
            .flat_map(|s| s.vertex_iter())
            .collect()
    }

    /// Check membership.
    #[inline]
    pub fn contains(&self, s: &Simplex) -> bool {
        self.membership.contains(s)
    }

    /// Star of a simplex σ: all simplices τ such that σ is a face of τ.
    pub fn star(&self, sigma: &Simplex) -> Vec<&Simplex> {
        self.all
            .iter()
            .filter(|tau| sigma.is_face_of(tau))
            .collect()
    }

    /// Link of a simplex σ: faces of St(σ) that don't intersect σ.
    pub fn link(&self, sigma: &Simplex) -> Vec<Simplex> {
        let star = self.star(sigma);
        let mut result: BTreeSet<Simplex> = BTreeSet::new();
        for tau in &star {
            for (_, face) in tau.faces() {
                if face.is_disjoint(sigma) && self.membership.contains(&face) {
                    result.insert(face);
                }
            }
        }
        result.into_iter().collect()
    }

    /// Compute boundary matrix ∂_k : C_k → C_{k-1} over Z/2Z.
    pub fn boundary_matrix_z2(&self, k: isize) -> BoundaryMatrix {
        let k_simplices = self.simplices_of_dim(k);
        let k_minus_1_simplices = self.simplices_of_dim(k - 1);

        // Index map: (k-1)-simplex → row index.
        let row_index: HashMap<&Simplex, usize> = k_minus_1_simplices
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i))
            .collect();

        let rows = k_minus_1_simplices.len();
        let cols = k_simplices.len();
        let faces_per = if cols > 0 { k_simplices[0].len() } else { 0 };
        let mut entries: Vec<(usize, usize)> = Vec::with_capacity(faces_per * cols);

        for (j, sigma) in k_simplices.iter().enumerate() {
            for (_, face) in sigma.faces() {
                if let Some(&i) = row_index.get(&face) {
                    entries.push((i, j));
                }
            }
        }

        BoundaryMatrix {
            rows,
            cols,
            entries,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Simplex> {
        self.all.iter()
    }

    /// Compute β₀ (connected components) via union-find in O(n·α(n)).
    /// Far faster than boundary matrix reduction for this specific invariant.
    pub fn connected_components(&self) -> usize {
        let verts = self.vertices();
        if verts.is_empty() {
            return 0;
        }
        let max_v = *verts.iter().max().unwrap();
        let mut uf = UnionFind::new(max_v + 1);
        for edge in self.simplices_of_dim(1) {
            let v: Vec<usize> = edge.vertex_iter().collect();
            uf.union(v[0], v[1]);
        }
        let mut roots: BTreeSet<usize> = BTreeSet::new();
        for &v in &verts {
            roots.insert(uf.find(v));
        }
        roots.len()
    }

    /// Build a Vietoris-Rips complex from a point cloud at a given radius.
    ///
    /// Algorithmic tricks:
    /// 1. Flat upper-triangle distance² matrix — no sqrt, no Vec<Vec<>>.
    /// 2. Sorted forward-adjacency lists for clique enumeration.
    /// 3. Merge-intersection for triangle/tet discovery.
    /// 4. Direct simplex insertion bypassing closure (faces are inserted
    ///    explicitly at each dimension level).
    pub fn vietoris_rips(points: &[[f64; 2]], radius: f64) -> Self {
        let n = points.len();
        let mut complex = Self::new();
        if n == 0 {
            return complex;
        }

        let r_sq = radius * radius;

        // Flat upper-triangle squared distance matrix.
        let tri_size = n * (n - 1) / 2;
        let mut dist_sq: Vec<f64> = Vec::with_capacity(tri_size);
        for i in 0..n {
            for j in (i + 1)..n {
                dist_sq.push(
                    (points[i][0] - points[j][0]).powi(2) + (points[i][1] - points[j][1]).powi(2),
                );
            }
        }
        let dsq = |i: usize, j: usize| -> f64 {
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            dist_sq[a * n - a * (a + 1) / 2 + b - a - 1]
        };

        // 0-simplices: all vertices.
        for i in 0..n {
            complex.insert_raw(Simplex::from_sorted(vec![i]), 0);
        }

        // 1-simplices: edges. Build sorted forward-adjacency lists.
        let mut adj_fwd: Vec<Vec<usize>> = vec![vec![]; n];
        // i is a vertex index used for distance lookup and simplex construction,
        // not just for indexing adj_fwd, so a range loop is appropriate here.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in (i + 1)..n {
                if dsq(i, j) <= r_sq {
                    complex.insert_raw(Simplex::from_sorted(vec![i, j]), 1);
                    adj_fwd[i].push(j);
                }
            }
        }
        // adj_fwd[i] is already sorted because j iterates in order.

        // 2-simplices: triangles via sorted merge-intersection.
        for i in 0..n {
            for idx_j in 0..adj_fwd[i].len() {
                let j = adj_fwd[i][idx_j];
                // Find k > j in adj_fwd[i] ∩ adj_fwd[j].
                let i_start = idx_j + 1; // adj_fwd[i] elements after j
                let mut pi = i_start;
                let mut pj = 0;
                while pi < adj_fwd[i].len() && pj < adj_fwd[j].len() {
                    let ki = adj_fwd[i][pi];
                    let kj = adj_fwd[j][pj];
                    match ki.cmp(&kj) {
                        std::cmp::Ordering::Equal => {
                            complex.insert_raw(Simplex::from_sorted(vec![i, j, ki]), 2);
                            pi += 1;
                            pj += 1;
                        }
                        std::cmp::Ordering::Less => pi += 1,
                        std::cmp::Ordering::Greater => pj += 1,
                    }
                }
            }
        }

        // 3-simplices: tetrahedra via triple merge-intersection.
        for i in 0..n {
            for idx_j in 0..adj_fwd[i].len() {
                let j = adj_fwd[i][idx_j];
                let i_after_j = idx_j + 1;
                for pi_k in i_after_j..adj_fwd[i].len() {
                    let k = adj_fwd[i][pi_k];
                    if dsq(j, k) > r_sq {
                        continue;
                    }
                    // Find l > k in adj_fwd[i] ∩ adj_fwd[j] ∩ adj_fwd[k].
                    let mut pi = pi_k + 1;
                    let mut pj = adj_fwd[j].partition_point(|&x| x <= k);
                    let mut pk = 0usize;
                    while pi < adj_fwd[i].len() && pj < adj_fwd[j].len() && pk < adj_fwd[k].len() {
                        let li = adj_fwd[i][pi];
                        let lj = adj_fwd[j][pj];
                        let lk = adj_fwd[k][pk];
                        let m = li.max(lj).max(lk);
                        if li == m && lj == m && lk == m {
                            complex.insert_raw(Simplex::from_sorted(vec![i, j, k, m]), 3);
                            pi += 1;
                            pj += 1;
                            pk += 1;
                        } else {
                            if li < m {
                                pi += 1;
                            }
                            if lj < m {
                                pj += 1;
                            }
                            if lk < m {
                                pk += 1;
                            }
                        }
                    }
                }
            }
        }

        complex
    }

    /// Internal: insert a simplex that is known to be at a given dimension,
    /// bypassing face-closure (caller guarantees faces are already present).
    fn insert_raw(&mut self, s: Simplex, dim: usize) {
        if self.membership.contains(&s) {
            return;
        }
        if dim as isize > self.max_dim {
            self.max_dim = dim as isize;
        }
        while self.by_dim.len() <= dim {
            self.by_dim.push(Vec::new());
        }
        let idx = self.all.len();
        self.by_dim[dim].push(idx);
        self.membership.insert(s.clone());
        self.all.push(s);
    }
}

impl Default for SimplicialComplex {
    fn default() -> Self {
        Self::new()
    }
}

/// Union-Find (disjoint set) with path compression and union by rank.
/// Used for O(n·α(n)) connected component computation.
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // path compression
        }
        self.parent[x]
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        // Union by rank.
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

/// Sparse boundary matrix over Z/2Z.
#[derive(Debug, Clone)]
pub struct BoundaryMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize)>,
}

impl BoundaryMatrix {
    /// Compute the rank via column reduction over Z/2Z.
    /// Columns packed into u64 bitsets for 64x XOR throughput.
    pub fn rank_z2(&self) -> usize {
        if self.rows == 0 || self.cols == 0 {
            return 0;
        }

        let wpc = self.rows.div_ceil(64);
        let mut cols: Vec<Vec<u64>> = vec![vec![0u64; wpc]; self.cols];
        for &(r, c) in &self.entries {
            cols[c][r / 64] |= 1u64 << (r % 64);
        }

        let low_of = |col: &[u64]| -> Option<usize> {
            for w in (0..col.len()).rev() {
                if col[w] != 0 {
                    return Some(w * 64 + (63 - col[w].leading_zeros() as usize));
                }
            }
            None
        };

        let mut pivot_col: Vec<Option<usize>> = vec![None; self.rows];
        let mut rank = 0;

        for j in 0..self.cols {
            loop {
                match low_of(&cols[j]) {
                    None => break,
                    Some(r) => {
                        if let Some(prev) = pivot_col[r] {
                            // XOR: 64 rows per operation.
                            let prev_col = cols[prev].clone();
                            for (w, pw) in cols[j].iter_mut().zip(prev_col.iter()) {
                                *w ^= pw;
                            }
                        } else {
                            pivot_col[r] = Some(j);
                            rank += 1;
                            break;
                        }
                    }
                }
            }
        }

        rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_complex() {
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1, 2]);
        assert_eq!(k.dimension(), 2);
        assert_eq!(k.f_vector(), vec![3, 3, 1]);
        assert_eq!(k.euler_characteristic(), 1);
    }

    #[test]
    fn hollow_tetrahedron() {
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1, 2]);
        k.insert_simplex(&[0, 1, 3]);
        k.insert_simplex(&[0, 2, 3]);
        k.insert_simplex(&[1, 2, 3]);
        assert_eq!(k.f_vector(), vec![4, 6, 4]);
        assert_eq!(k.euler_characteristic(), 2);
    }

    #[test]
    fn vietoris_rips_triangle() {
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
        let complex = SimplicialComplex::vietoris_rips(&points, 1.1);
        assert_eq!(complex.f_vector(), vec![3, 3, 1]);
    }

    #[test]
    fn union_find_components() {
        let mut k = SimplicialComplex::new();
        k.insert_simplex(&[0, 1]);
        k.insert_simplex(&[2, 3]);
        assert_eq!(k.connected_components(), 2);

        k.insert_simplex(&[1, 2]);
        assert_eq!(k.connected_components(), 1);
    }

    #[test]
    fn vr_empty() {
        let complex = SimplicialComplex::vietoris_rips(&[], 1.0);
        assert_eq!(complex.size(), 0);
    }

    #[test]
    fn vr_single_point() {
        let complex = SimplicialComplex::vietoris_rips(&[[0.0, 0.0]], 1.0);
        assert_eq!(complex.f_vector(), vec![1]);
    }
}
