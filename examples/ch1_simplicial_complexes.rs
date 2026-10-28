//! Chapter 1 Worked Examples: Simplicial Complexes
//!
//! Run with: cargo run --example ch1_simplicial_complexes

use topo_sonify::topology::complex::SimplicialComplex;
use topo_sonify::topology::homology::BettiNumbers;
use topo_sonify::topology::simplex::Simplex;

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  Chapter 1: Simplicial Complexes — Worked Examples");
    println!("═══════════════════════════════════════════════════════\n");

    exercise_1_1();
    exercise_1_2();
    exercise_1_3();
    exercise_1_4();
    exercise_1_5();
    exercise_1_6();
}

/// Exercise 1.1: Build a triangle and inspect its structure.
///
/// We insert the single 2-simplex [0,1,2]. The closure property
/// automatically generates all faces: edges [0,1], [0,2], [1,2]
/// and vertices [0], [1], [2].
fn exercise_1_1() {
    println!("── Exercise 1.1: The closure property ──────────────────");
    println!("Insert one 2-simplex [0,1,2] into an empty complex.\n");

    let mut k = SimplicialComplex::new();
    k.insert_simplex(&[0, 1, 2]);

    println!(
        "  Total simplices: {}  (we inserted 1, closure added 6)",
        k.size()
    );
    println!("  f-vector: {:?}", k.f_vector());
    println!("  Euler characteristic: χ = {}", k.euler_characteristic());
    println!("  Dimension: {}", k.dimension());

    // Walk through each simplex.
    println!("\n  All simplices in the complex:");
    for s in k.iter() {
        let dim = s.dimension();
        let name = match dim {
            0 => "vertex  ",
            1 => "edge    ",
            2 => "triangle",
            _ => "simplex ",
        };
        println!("    {} (dim {}) : {:?}", name, dim, s.vertices());
    }

    // Verify: f = (3, 3, 1), χ = 3 - 3 + 1 = 1.
    assert_eq!(k.f_vector(), vec![3, 3, 1]);
    assert_eq!(k.euler_characteristic(), 1);
    println!("\n  ✓ f = (3,3,1), χ = 1. A filled triangle is contractible.\n");
}

/// Exercise 1.2: The hollow tetrahedron (boundary of Δ³ ≅ S²).
///
/// Four triangular faces, but no solid tetrahedron. The result is
/// homeomorphic to the 2-sphere S². We verify χ(S²) = 2.
fn exercise_1_2() {
    println!("── Exercise 1.2: The hollow tetrahedron (S²) ───────────");

    let mut k = SimplicialComplex::new();
    k.insert_simplex(&[0, 1, 2]);
    k.insert_simplex(&[0, 1, 3]);
    k.insert_simplex(&[0, 2, 3]);
    k.insert_simplex(&[1, 2, 3]);

    println!("  f-vector: {:?}", k.f_vector());
    println!(
        "  χ = {} - {} + {} = {}",
        k.f_vector()[0],
        k.f_vector()[1],
        k.f_vector()[2],
        k.euler_characteristic()
    );

    let betti = BettiNumbers::compute(&k);
    println!("  Betti numbers: β = {:?}", betti.values);
    println!("  β₀ = {} (one component)", betti.b0());
    println!(
        "  β₁ = {} (no loops — every loop on S² bounds a disk)",
        betti.b1()
    );
    println!("  β₂ = {} (one void — the enclosed cavity)", betti.b2());

    // Euler-Poincaré: χ = β₀ - β₁ + β₂ = 1 - 0 + 1 = 2.
    assert_eq!(betti.euler_characteristic(), 2);
    println!("  ✓ χ = 1 - 0 + 1 = 2. This is the sphere S².\n");
}

/// Exercise 1.3: Vietoris-Rips complex at varying radii.
///
/// Place 4 points in a unit square and sweep the radius parameter.
/// Watch how the topology changes:
///   r < 1.0  : edges form between adjacent corners
///   r ≈ 1.0  : all side edges present, creating a cycle (β₁ = 1)
///   r ≈ 1.41 : diagonal edges appear, filling the loop (β₁ = 0)
fn exercise_1_3() {
    println!("── Exercise 1.3: VR complex of a unit square ───────────");
    let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    let radii = [0.5, 0.9, 1.05, 1.45, 2.0];
    println!("  Points: (0,0), (1,0), (1,1), (0,1)\n");
    println!(
        "  {:>6} │ {:>12} │ {:>4} │ {:>12} │ topology",
        "radius", "f-vector", "χ", "β"
    );
    println!("  ───────┼──────────────┼──────┼──────────────┼─────────────────");

    for &r in &radii {
        let complex = SimplicialComplex::vietoris_rips(&points, r);
        let betti = BettiNumbers::compute(&complex);
        let f = complex.f_vector();
        let chi = complex.euler_characteristic();

        let topo = if betti.b0() > 1 {
            format!("{} components", betti.b0())
        } else if betti.b1() > 0 {
            format!("{} loop(s)!", betti.b1())
        } else {
            "contractible".to_string()
        };

        println!(
            "  {:>6.2} │ {:>12} │ {:>4} │ {:>12} │ {}",
            r,
            format!("{:?}", f),
            chi,
            format!("{:?}", betti.values),
            topo
        );
    }
    println!();
    println!("  Key insight: at r ≈ 1.05, all 4 side edges exist but no");
    println!("  diagonals — creating a 1-cycle. At r ≈ 1.45, the diagonals");
    println!("  appear and fill the loop. The loop is born and then killed.\n");
}

/// Exercise 1.4: Face enumeration and the boundary.
///
/// A tetrahedron [0,1,2,3] has 4 triangular faces.
/// Each face is obtained by removing one vertex.
/// The alternating signs (-1)^i give the oriented boundary.
fn exercise_1_4() {
    println!("── Exercise 1.4: Faces of a tetrahedron ────────────────");
    let tet = Simplex::new([0, 1, 2, 3]);
    println!(
        "  Tetrahedron σ = {:?}, dim = {}\n",
        tet.vertices(),
        tet.dimension()
    );

    println!("  Faces (obtained by removing one vertex):");
    for (i, face) in tet.faces() {
        let sign = if i % 2 == 0 { "+" } else { "-" };
        println!(
            "    i={}: remove v_{} → {} {:?}",
            i,
            tet.vertices()[i],
            sign,
            face.vertices()
        );
    }
    println!("\n  The boundary is: ∂σ = +[1,2,3] - [0,2,3] + [0,1,3] - [0,1,2]");
    println!("  Over Z/2Z (mod 2), all signs become +1.\n");
}

/// Exercise 1.5: The f-vector determines sonification ratios.
///
/// The BettiDrone uses f_{k+1}/f_k as just-intonation frequency ratios.
/// Different complexes produce different tunings.
fn exercise_1_5() {
    println!("── Exercise 1.5: f-vector → frequency ratios ───────────");

    let complexes: Vec<(&str, Vec<Vec<usize>>)> = vec![
        ("Filled triangle", vec![vec![0, 1, 2]]),
        (
            "Hollow tetrahedron (S²)",
            vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]],
        ),
        (
            "Two triangles sharing an edge",
            vec![vec![0, 1, 2], vec![0, 1, 3]],
        ),
        (
            "Octahedron (S²)",
            vec![
                vec![0, 1, 2],
                vec![0, 2, 3],
                vec![0, 3, 4],
                vec![0, 4, 1],
                vec![5, 1, 2],
                vec![5, 2, 3],
                vec![5, 3, 4],
                vec![5, 4, 1],
            ],
        ),
    ];

    for (name, simplices) in &complexes {
        let mut k = SimplicialComplex::new();
        for s in simplices {
            k.insert_simplex(s);
        }
        let f = k.f_vector();
        let ratios: Vec<String> = f
            .windows(2)
            .map(|w| format!("{}/{} = {:.3}", w[1], w[0], w[1] as f64 / w[0] as f64))
            .collect();

        println!("  {}", name);
        println!("    f = {:?}", f);
        println!("    Ratios: {}", ratios.join(", "));

        // Show what interval these ratios approximate.
        for w in f.windows(2) {
            let r = w[1] as f64 / w[0] as f64;
            let interval = if (r - 1.0).abs() < 0.1 {
                "unison"
            } else if (r - 1.2).abs() < 0.1 {
                "minor third"
            } else if (r - 1.25).abs() < 0.1 {
                "major third"
            } else if (r - 1.333).abs() < 0.1 {
                "fourth"
            } else if (r - 1.5).abs() < 0.1 {
                "fifth"
            } else if (r - 1.667).abs() < 0.1 {
                "minor sixth"
            } else if (r - 2.0).abs() < 0.15 {
                "octave"
            } else {
                "non-standard"
            };
            println!(
                "    f_{}/f_{} ≈ {:.3} → {}",
                f.len().min(2),
                f.len().min(1),
                r,
                interval
            );
        }
        println!();
    }
}

/// Exercise 1.6: Connected components via union-find vs. matrix reduction.
///
/// Demonstrates that β₀ can be computed two ways — and that union-find
/// is dramatically faster for large complexes.
fn exercise_1_6() {
    println!("── Exercise 1.6: Two paths to β₀ ──────────────────────");
    let mut k = SimplicialComplex::new();
    k.insert_simplex(&[0, 1]);
    k.insert_simplex(&[1, 2]);
    k.insert_simplex(&[3, 4]); // separate component
    k.insert_simplex(&[5]); // isolated vertex

    println!("  Complex: edges [0,1], [1,2], [3,4], vertex [5]");
    println!(
        "  Via union-find:      {} components",
        k.connected_components()
    );
    let betti = BettiNumbers::compute(&k);
    println!("  Via rank-nullity:    β₀ = {}", betti.b0());
    assert_eq!(k.connected_components(), betti.b0());
    println!("  ✓ Both agree: 3 components.\n");
    println!("  Union-find is O(n·α(n)); matrix reduction is O(n³).");
    println!("  For β₀ alone, always use union-find.\n");
}
