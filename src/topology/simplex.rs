use serde::{Deserialize, Serialize};

/// A k-simplex: a sorted set of (k+1) vertex indices.
///
/// Stored as a sorted `Vec<usize>` rather than `BTreeSet` for:
/// - Cache-friendly contiguous memory (no tree node allocations)
/// - O(k) hashing and comparison (not O(k log k) tree traversal)
/// - Efficient face enumeration without per-face allocation overhead
///
/// Invariant: `vertices` is always sorted and deduplicated.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Simplex {
    vertices: Vec<usize>,
}

impl Simplex {
    /// Create a simplex from an iterator of vertices.
    /// Sorts and deduplicates automatically.
    pub fn new(vertices: impl IntoIterator<Item = usize>) -> Self {
        let mut v: Vec<usize> = vertices.into_iter().collect();
        v.sort_unstable();
        v.dedup();
        Self { vertices: v }
    }

    /// Create from an already-sorted, deduplicated slice. No validation.
    /// Caller must guarantee the slice is sorted and unique.
    pub(crate) fn from_sorted(vertices: Vec<usize>) -> Self {
        Self { vertices }
    }

    /// The dimension of the simplex: |vertices| - 1
    #[inline]
    pub fn dimension(&self) -> isize {
        self.vertices.len() as isize - 1
    }

    /// Compute all (k-1)-dimensional faces by removing one vertex at a time.
    /// Returns pairs of (removed_vertex_position, face).
    ///
    /// Each face is constructed by skipping one element from the sorted array,
    /// so the result is automatically sorted — no re-sorting needed.
    pub fn faces(&self) -> Vec<(usize, Simplex)> {
        let n = self.vertices.len();
        (0..n)
            .map(|i| {
                let mut face_verts = Vec::with_capacity(n - 1);
                face_verts.extend_from_slice(&self.vertices[..i]);
                face_verts.extend_from_slice(&self.vertices[i + 1..]);
                (i, Simplex::from_sorted(face_verts))
            })
            .collect()
    }

    /// Check if this simplex is a face of another (subset relationship on sorted vecs).
    pub fn is_face_of(&self, other: &Simplex) -> bool {
        if self.vertices.len() > other.vertices.len() {
            return false;
        }
        // Merge-style subset check on two sorted arrays: O(|self| + |other|).
        let mut j = 0;
        for &v in &self.vertices {
            while j < other.vertices.len() && other.vertices[j] < v {
                j += 1;
            }
            if j >= other.vertices.len() || other.vertices[j] != v {
                return false;
            }
            j += 1;
        }
        true
    }

    /// The number of vertices.
    #[inline]
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Vertex iterator.
    pub fn vertex_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.vertices.iter().copied()
    }

    /// Direct access to the vertex slice.
    #[inline]
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }

    /// Check if two simplices share no vertices (for link computation).
    pub fn is_disjoint(&self, other: &Simplex) -> bool {
        // Merge-style on sorted arrays.
        let (mut i, mut j) = (0, 0);
        while i < self.vertices.len() && j < other.vertices.len() {
            match self.vertices[i].cmp(&other.vertices[j]) {
                std::cmp::Ordering::Equal => return false,
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        true
    }
}

impl From<Vec<usize>> for Simplex {
    fn from(v: Vec<usize>) -> Self {
        Self::new(v)
    }
}

impl From<&[usize]> for Simplex {
    fn from(s: &[usize]) -> Self {
        Self::new(s.iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_dimension() {
        assert_eq!(Simplex::new([0]).dimension(), 0);
        assert_eq!(Simplex::new([0, 1]).dimension(), 1);
        assert_eq!(Simplex::new([0, 1, 2]).dimension(), 2);
        assert_eq!(Simplex::new([0, 1, 2, 3]).dimension(), 3);
    }

    #[test]
    fn triangle_faces() {
        let tri = Simplex::new([0, 1, 2]);
        let faces = tri.faces();
        assert_eq!(faces.len(), 3);
        let face_verts: Vec<Vec<usize>> = faces
            .into_iter()
            .map(|(_, f)| f.vertices().to_vec())
            .collect();
        assert!(face_verts.contains(&vec![1, 2]));
        assert!(face_verts.contains(&vec![0, 2]));
        assert!(face_verts.contains(&vec![0, 1]));
    }

    #[test]
    fn face_relationship() {
        let edge = Simplex::new([0, 1]);
        let tri = Simplex::new([0, 1, 2]);
        assert!(edge.is_face_of(&tri));
        assert!(!tri.is_face_of(&edge));
    }

    #[test]
    fn is_disjoint() {
        let a = Simplex::new([0, 1]);
        let b = Simplex::new([2, 3]);
        let c = Simplex::new([1, 2]);
        assert!(a.is_disjoint(&b));
        assert!(!a.is_disjoint(&c));
    }

    #[test]
    fn deduplication() {
        let s = Simplex::new([2, 1, 2, 0, 1]);
        assert_eq!(s.vertices(), &[0, 1, 2]);
    }
}
