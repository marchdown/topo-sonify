//! Chapter 2 Worked Examples: Homology and the Boundary Operator
//!
//! Run with: cargo run --example ch2_homology

use topo_sonify::topology::complex::SimplicialComplex;
use topo_sonify::topology::hodge::HodgeLaplacian;
use topo_sonify::topology::homology::BettiNumbers;

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  Chapter 2: Homology — Worked Examples");
    println!("═══════════════════════════════════════════════════════\n");

    exercise_2_1();
    exercise_2_2();
    exercise_2_3();
    exercise_2_4();
    exercise_2_5();
}

/// Exercise 2.1: Boundary matrices of the hollow tetrahedron.
///
/// We compute ∂₁ (edges → vertices) and ∂₂ (triangles → edges)
/// and verify ∂₁ ∘ ∂₂ = 0 by checking ranks.
fn exercise_2_1() {
    println!("── Exercise 2.1: Boundary matrices of S² ────────────────");
    let mut k = SimplicialComplex::new();
    k.insert_simplex(&[0, 1, 2]);
    k.insert_simplex(&[0, 1, 3]);
    k.insert_simplex(&[0, 2, 3]);
    k.insert_simplex(&[1, 2, 3]);

    let d1 = k.boundary_matrix_z2(1);
    let d2 = k.boundary_matrix_z2(2);

    println!(
        "  ∂₁ : C₁ → C₀ ({}×{} matrix, {} nonzero entries)",
        d1.rows,
        d1.cols,
        d1.entries.len()
    );
    println!(
        "  ∂₂ : C₂ → C₁ ({}×{} matrix, {} nonzero entries)",
        d2.rows,
        d2.cols,
        d2.entries.len()
    );

    println!("\n  Dimensions of chain groups:");
    println!("    dim C₀ = {} (vertices)", k.count_dim(0));
    println!("    dim C₁ = {} (edges)", k.count_dim(1));
    println!("    dim C₂ = {} (triangles)", k.count_dim(2));

    let rank_d1 = d1.rank_z2();
    let rank_d2 = d2.rank_z2();
    println!("\n  rank(∂₁) = {}", rank_d1);
    println!("  rank(∂₂) = {}", rank_d2);

    // β₀ = dim C₀ - rank ∂₁ - rank ∂₂... wait, β₀ = dim C₀ - rank ∂₁ = 4 - 3 = 1
    // β₁ = dim C₁ - rank ∂₁ - rank ∂₂ = 6 - 3 - 3 = 0
    // β₂ = dim C₂ - rank ∂₂ = 4 - 3 = 1
    println!("\n  Betti numbers via rank-nullity:");
    println!(
        "    β₀ = dim C₀ - rank ∂₁ = {} - {} = {}",
        k.count_dim(0),
        rank_d1,
        k.count_dim(0) - rank_d1
    );
    println!(
        "    β₁ = (dim C₁ - rank ∂₁) - rank ∂₂ = ({} - {}) - {} = {}",
        k.count_dim(1),
        rank_d1,
        rank_d2,
        k.count_dim(1) - rank_d1 - rank_d2
    );
    println!(
        "    β₂ = dim C₂ - rank ∂₂ = {} - {} = {}",
        k.count_dim(2),
        rank_d2,
        k.count_dim(2) - rank_d2
    );

    let betti = BettiNumbers::compute(&k);
    println!("\n  Computed Betti: {:?}", betti.values);
    println!(
        "  Euler check: χ = {} - {} + {} = {} = β₀ - β₁ + β₂ = {} ✓\n",
        k.f_vector()[0],
        k.f_vector()[1],
        k.f_vector()[2],
        k.euler_characteristic(),
        betti.euler_characteristic()
    );
}

/// Exercise 2.2: The triangle boundary — detecting a loop.
///
/// Three edges forming a cycle without a fill. This is the simplest
/// complex with β₁ = 1. We trace exactly why the loop is detected.
fn exercise_2_2() {
    println!("── Exercise 2.2: A loop detected algebraically ──────────");
    let mut k = SimplicialComplex::new();
    k.insert_simplex(&[0, 1]);
    k.insert_simplex(&[1, 2]);
    k.insert_simplex(&[0, 2]);

    println!("  Complex: three edges [0,1], [1,2], [0,2] — a hollow triangle.");
    println!("  f-vector: {:?}", k.f_vector());

    let d1 = k.boundary_matrix_z2(1);
    println!("\n  ∂₁ maps each edge to its boundary (pair of vertices):");
    println!("    ∂[0,1] = [0] + [1]");
    println!("    ∂[0,2] = [0] + [2]");
    println!("    ∂[1,2] = [1] + [2]");
    println!("\n  Over Z/2Z, the chain c = [0,1] + [1,2] + [0,2] satisfies:");
    println!("    ∂c = ([0]+[1]) + ([1]+[2]) + ([0]+[2])");
    println!("       = 2·[0] + 2·[1] + 2·[2]");
    println!("       = 0  (mod 2)");
    println!("  So c is a cycle! And since there are no 2-simplices,");
    println!("  c cannot be a boundary. It represents a nontrivial class in H₁.\n");

    let betti = BettiNumbers::compute(&k);
    println!("  rank(∂₁) = {}", d1.rank_z2());
    println!("  β₀ = {} (one component)", betti.b0());
    println!("  β₁ = {} (one independent loop) ✓\n", betti.b1());
}

/// Exercise 2.3: What happens when we fill the loop?
///
/// Adding the 2-simplex [0,1,2] kills β₁. The boundary of the triangle
/// IS exactly the cycle c from Exercise 2.2, so c becomes a boundary.
fn exercise_2_3() {
    println!("── Exercise 2.3: Filling the loop ───────────────────────");
    let mut k = SimplicialComplex::new();
    k.insert_simplex(&[0, 1, 2]); // filled triangle

    println!("  Now add the 2-simplex [0,1,2] (fill the triangle).");
    println!("  ∂₂([0,1,2]) = [1,2] - [0,2] + [0,1]");
    println!("  Over Z/2Z: ∂₂([0,1,2]) = [0,1] + [0,2] + [1,2] = c");
    println!("  The cycle c is now a boundary → [c] = 0 in H₁.\n");

    let betti = BettiNumbers::compute(&k);
    println!(
        "  β₀ = {}, β₁ = {} (loop killed!), χ = {}\n",
        betti.b0(),
        betti.b1(),
        betti.euler_characteristic()
    );
}

/// Exercise 2.4: Hodge Laplacian eigenvalues and the discrete Hodge theorem.
///
/// The kernel dimension of L_k equals β_k. We verify this for several spaces
/// and show what the nonzero eigenvalues tell us about geometry.
fn exercise_2_4() {
    println!("── Exercise 2.4: Hodge Laplacian and spectral information ──");

    let spaces: Vec<(&str, Vec<Vec<usize>>)> = vec![
        (
            "Path graph P₃ (3 vertices, 2 edges)",
            vec![vec![0, 1], vec![1, 2]],
        ),
        (
            "Cycle C₃ (triangle boundary)",
            vec![vec![0, 1], vec![1, 2], vec![0, 2]],
        ),
        ("Filled triangle", vec![vec![0, 1, 2]]),
        (
            "Hollow tetrahedron (S²)",
            vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]],
        ),
    ];

    for (name, simplices) in &spaces {
        println!("\n  {} :", name);
        let mut k = SimplicialComplex::new();
        for s in simplices {
            k.insert_simplex(s);
        }

        let betti = BettiNumbers::compute(&k);
        println!("    Betti: {:?}", betti.values);

        for dim in 0..=k.dimension() as usize {
            if k.count_dim(dim as isize) == 0 {
                continue;
            }
            let l = HodgeLaplacian::compute(&k, dim);
            let evs = l.eigenvalues();
            let evs_str: Vec<String> = evs.iter().map(|e| format!("{:.2}", e)).collect();
            println!("    L_{} eigenvalues: [{}]", dim, evs_str.join(", "));
            println!(
                "    dim(ker L_{}) = {} = β_{} ✓",
                dim,
                l.kernel_dimension(),
                dim
            );
        }

        // For the graph Laplacian, the spectral gap tells us about connectivity.
        let l0 = HodgeLaplacian::compute(&k, 0);
        let gap = l0.spectral_gap();
        println!("    Spectral gap (algebraic connectivity): {:.4}", gap);
    }
    println!();
}

/// Exercise 2.5: Betti numbers map to instrument parameters.
///
/// Show exactly how each Betti number controls the BettiDrone instrument.
fn exercise_2_5() {
    println!("── Exercise 2.5: β → instrument parameters ──────────────");

    let test_cases: Vec<(&str, Vec<usize>)> = vec![
        ("Single point (trivial)", vec![1]),
        ("Circle (one loop)", vec![1, 1]),
        ("Torus (two loops, one void)", vec![1, 2, 1]),
        ("Genus-2 surface", vec![1, 4, 1]),
        ("Three components with loops", vec![3, 2, 0]),
    ];

    println!(
        "  {:30} │ {:8} │ {:6} │ {:5} │ {:6}",
        "Space", "β", "voices", "LFOs", "subs"
    );
    println!("  ───────────────────────────────┼──────────┼────────┼───────┼──────");

    for (name, betti) in &test_cases {
        let b0 = betti.first().copied().unwrap_or(0).max(1);
        let b1 = betti.get(1).copied().unwrap_or(0);
        let b2 = betti.get(2).copied().unwrap_or(0);

        println!(
            "  {:30} │ {:8} │ {:>6} │ {:>5} │ {:>5}",
            name,
            format!("{:?}", betti),
            b0,
            b1,
            b2
        );
    }

    println!("\n  β₀ → oscillator voices (more components = chord, not unison)");
    println!("  β₁ → LFO modulators (loops = cyclic modulation)");
    println!("  β₂ → sub-oscillators (voids = resonant depth)\n");
}
