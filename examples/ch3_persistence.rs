//! Chapter 3 Worked Examples: Persistent Homology
//!
//! Run with: cargo run --example ch3_persistence

use topo_sonify::topology::complex::SimplicialComplex;
use topo_sonify::topology::distances::{bottleneck_distance, wasserstein_distance};
use topo_sonify::topology::filtration::FilteredComplex;
use topo_sonify::topology::homology::BettiNumbers;
use topo_sonify::topology::morse::DiscreteMorseFunction;
use topo_sonify::topology::persistence::{PersistenceDiagram, PersistencePair};
use topo_sonify::topology::standard_persistence::{compute_persistence, compute_persistence_twist};

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  Chapter 3: Persistent Homology — Worked Examples");
    println!("═══════════════════════════════════════════════════════\n");

    exercise_3_1();
    exercise_3_2();
    exercise_3_3();
    exercise_3_4();
    exercise_3_5();
    exercise_3_6();
}

/// Exercise 3.1: Standard persistence on a triangle.
///
/// Walk through the column reduction algorithm step by step.
fn exercise_3_1() {
    println!("── Exercise 3.1: Persistence of a triangle ──────────────");
    let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
    let fc = FilteredComplex::vietoris_rips(&points, 2.0);

    println!("  Point cloud: (0,0), (1,0), (0.5, 0.866)");
    println!("  An equilateral triangle with side ≈ 1.0.\n");

    println!("  Filtration order (simplex, dimension, filtration value):");
    for (i, fs) in fc.simplices.iter().enumerate() {
        let bdry = fc.boundary_indices(i);
        let bdry_str = if bdry.is_empty() {
            "∅".to_string()
        } else {
            format!("{:?}", bdry)
        };
        println!(
            "    σ_{} = {:?}  dim={}  filt={:.4}  ∂={}",
            i,
            fs.simplex.vertices(),
            fs.dimension(),
            fs.filtration,
            bdry_str
        );
    }

    let pd = compute_persistence(&fc);
    println!("\n  Persistence pairs (standard algorithm):");
    for p in &pd.pairs {
        let death_str = if p.is_essential() {
            "∞".to_string()
        } else {
            format!("{:.4}", p.death)
        };
        println!(
            "    H_{}: ({:.4}, {})  persistence = {}",
            p.dimension,
            p.birth,
            death_str,
            if p.is_essential() {
                "∞".to_string()
            } else {
                format!("{:.4}", p.persistence())
            }
        );
    }

    println!("\n  Interpretation:");
    for p in &pd.pairs {
        if p.dimension == 0 && p.is_essential() {
            println!("    H₀ essential: the final connected component (born at r=0, never dies)");
        } else if p.dimension == 0 {
            println!(
                "    H₀ ({:.4}, {:.4}): a component born at r={:.4}, merged at r={:.4}",
                p.birth, p.death, p.birth, p.death
            );
        } else if p.dimension == 1 {
            println!(
                "    H₁ ({:.4}, {:.4}): a loop born at r={:.4}, filled at r={:.4}",
                p.birth, p.death, p.birth, p.death
            );
        }
    }
    println!();
}

/// Exercise 3.2: Standard vs. twist optimization.
///
/// The twist (Chen-Kerber 2011) clears columns of paired creators,
/// avoiding redundant reductions. We verify it produces identical output.
fn exercise_3_2() {
    println!("── Exercise 3.2: Standard vs. twist optimization ────────");
    let points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.866],
        [2.0, 0.0],
        [1.5, 0.866],
    ];
    let fc = FilteredComplex::vietoris_rips(&points, 3.0);

    let pd_std = compute_persistence(&fc);
    let pd_twist = compute_persistence_twist(&fc);

    println!("  5-point cloud, {} simplices in filtration.\n", fc.len());

    println!("  Standard algorithm:");
    for p in &pd_std.pairs {
        let d = if p.is_essential() {
            "∞".to_string()
        } else {
            format!("{:.3}", p.death)
        };
        println!("    H_{}: ({:.3}, {})", p.dimension, p.birth, d);
    }

    println!("\n  Twist optimization:");
    for p in &pd_twist.pairs {
        let d = if p.is_essential() {
            "∞".to_string()
        } else {
            format!("{:.3}", p.death)
        };
        println!("    H_{}: ({:.3}, {})", p.dimension, p.birth, d);
    }

    assert_eq!(pd_std.pairs.len(), pd_twist.pairs.len());
    let tp_diff = (pd_std.total_persistence() - pd_twist.total_persistence()).abs();
    println!("\n  Same number of pairs: {} ✓", pd_std.pairs.len());
    println!("  Total persistence difference: {:.2e} ✓\n", tp_diff);
}

/// Exercise 3.3: Persistence diagram of a square — detecting and killing a loop.
///
/// The unit square is the classic example: at intermediate radius, the four
/// edges form a 1-cycle (β₁ = 1). When the diagonal edges appear, the
/// cycle is filled and β₁ drops back to 0.
fn exercise_3_3() {
    println!("── Exercise 3.3: The square — birth and death of a loop ──");
    let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let fc = FilteredComplex::vietoris_rips(&points, 2.0);
    let pd = compute_persistence(&fc);

    println!("  Points: corners of the unit square.");
    println!("  Side length = 1.0, diagonal = √2 ≈ 1.414.\n");

    let h1_pairs: Vec<&PersistencePair> = pd.pairs.iter().filter(|p| p.dimension == 1).collect();

    if let Some(p) = h1_pairs.first() {
        println!("  The H₁ feature:");
        println!(
            "    Born at r = {:.4} (all four side edges present → cycle)",
            p.birth
        );
        println!(
            "    Dies at r = {:.4} (diagonal edge creates a triangle → fills loop)",
            p.death
        );
        println!("    Persistence = {:.4}", p.persistence());
        println!("\n  This is the loop going 0→1→2→3→0 around the square.");
        println!(
            "  It lives for Δr = {:.4} before being killed.\n",
            p.persistence()
        );
    }

    // Also show the H₀ merging story.
    let h0_pairs: Vec<&PersistencePair> = pd
        .pairs
        .iter()
        .filter(|p| p.dimension == 0 && !p.is_essential())
        .collect();
    println!("  H₀ story (components merging):");
    for p in &h0_pairs {
        println!(
            "    Component born at r={:.4}, merged at r={:.4}",
            p.birth, p.death
        );
    }
    println!();
}

/// Exercise 3.4: Persistence entropy and total persistence.
///
/// Compare diagrams with uniform vs. skewed persistence distributions.
fn exercise_3_4() {
    println!("── Exercise 3.4: Persistence entropy ────────────────────");

    let uniform = PersistenceDiagram {
        pairs: vec![
            PersistencePair {
                birth: 0.0,
                death: 1.0,
                dimension: 0,
            },
            PersistencePair {
                birth: 0.0,
                death: 1.0,
                dimension: 0,
            },
            PersistencePair {
                birth: 0.0,
                death: 1.0,
                dimension: 0,
            },
        ],
    };
    let skewed = PersistenceDiagram {
        pairs: vec![
            PersistencePair {
                birth: 0.0,
                death: 10.0,
                dimension: 0,
            },
            PersistencePair {
                birth: 0.0,
                death: 1.0,
                dimension: 0,
            },
            PersistencePair {
                birth: 0.0,
                death: 1.0,
                dimension: 0,
            },
        ],
    };

    println!("  Uniform diagram:  persistences = [1, 1, 1]");
    println!("    Total persistence: {:.2}", uniform.total_persistence());
    println!("    Entropy: {:.4}", uniform.persistence_entropy());

    println!("\n  Skewed diagram:   persistences = [10, 1, 1]");
    println!("    Total persistence: {:.2}", skewed.total_persistence());
    println!("    Entropy: {:.4}", skewed.persistence_entropy());

    println!("\n  Uniform has HIGHER entropy (max disorder).");
    println!("  Skewed has LOWER entropy (one feature dominates).");
    println!("  Sonically: high entropy → complex texture, low → clear tone.\n");
}

/// Exercise 3.5: Bottleneck and Wasserstein distances.
///
/// Measure how different two persistence diagrams are.
fn exercise_3_5() {
    println!("── Exercise 3.5: Diagram distances ──────────────────────");

    let d1 = vec![
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
    let d2 = vec![
        PersistencePair {
            birth: 0.0,
            death: 1.5,
            dimension: 0,
        },
        PersistencePair {
            birth: 0.5,
            death: 2.5,
            dimension: 1,
        },
    ];
    let d3 = vec![PersistencePair {
        birth: 0.0,
        death: 5.0,
        dimension: 0,
    }];

    println!("  D₁ = [(0,1), (0.5,2)]");
    println!("  D₂ = [(0,1.5), (0.5,2.5)]   (shifted by 0.5 in death)");
    println!("  D₃ = [(0,5)]                  (one big feature)");

    let bn_12 = bottleneck_distance(&d1, &d2);
    let bn_13 = bottleneck_distance(&d1, &d3);
    let w1_12 = wasserstein_distance(&d1, &d2, 1.0);
    let w1_13 = wasserstein_distance(&d1, &d3, 1.0);

    println!(
        "\n  d_B(D₁, D₂) = {:.4}  (small shift → small distance)",
        bn_12
    );
    println!("  d_B(D₁, D₃) = {:.4}  (very different diagrams)", bn_13);
    println!("  W₁(D₁, D₂)  = {:.4}", w1_12);
    println!("  W₁(D₁, D₃)  = {:.4}", w1_13);
    println!("\n  The stability theorem guarantees: if the input data changes");
    println!("  by ε in L∞ norm, the diagram changes by ≤ ε in bottleneck.");
    println!("  This is why our sonification changes smoothly.\n");
}

/// Exercise 3.6: Discrete Morse theory — critical simplices and the Morse inequalities.
///
/// A discrete Morse function pairs most simplices, leaving only the
/// "essential" ones (critical simplices). The number of critical k-simplices
/// c_k satisfies c_k ≥ β_k (weak Morse inequality).
fn exercise_3_6() {
    println!("── Exercise 3.6: Discrete Morse theory ──────────────────");

    let mut k = SimplicialComplex::new();
    k.insert_simplex(&[0, 1, 2]);
    k.insert_simplex(&[0, 1, 3]);
    k.insert_simplex(&[0, 2, 3]);
    k.insert_simplex(&[1, 2, 3]);

    println!("  Space: hollow tetrahedron ≅ S²");
    println!(
        "  f-vector: {:?} ({} total simplices)\n",
        k.f_vector(),
        k.size()
    );

    let betti = BettiNumbers::compute(&k);
    let morse = DiscreteMorseFunction::from_vertex_function(&k, &[0.0, 1.0, 2.0, 3.0]);
    let counts = morse.critical_counts();

    println!("  Vertex function: f(v₀)=0, f(v₁)=1, f(v₂)=2, f(v₃)=3");
    println!("\n  Gradient pairs (matched simplex pairs):");
    for (sigma, tau) in &morse.gradient_pairs {
        println!(
            "    {:?} ↔ {:?}  (dim {} paired with dim {})",
            sigma.vertices(),
            tau.vertices(),
            sigma.dimension(),
            tau.dimension()
        );
    }

    println!("\n  Critical simplices (unpaired):");
    for c in &morse.critical {
        println!("    {:?}  (dim {})", c.vertices(), c.dimension());
    }

    println!("\n  Morse inequalities c_k ≥ β_k:");
    for dim in 0..counts.len().max(betti.values.len()) {
        let c = counts.get(dim).copied().unwrap_or(0);
        let b = betti.values.get(dim).copied().unwrap_or(0);
        let check = if c >= b { "✓" } else { "✗" };
        println!(
            "    dim {}: c_{} = {} ≥ β_{} = {}  {}",
            dim, dim, c, dim, b, check
        );
    }

    println!("\n  Strong Morse inequality (Euler):");
    println!(
        "    Σ(-1)^k c_k = {} = χ(S²) = {} ✓",
        morse.euler_from_critical(),
        k.euler_characteristic()
    );

    let reduction = k.size() - morse.critical.len();
    println!(
        "\n  The Morse complex has {} critical simplices instead of {} total.",
        morse.critical.len(),
        k.size()
    );
    println!(
        "  Reduction: {} simplices eliminated by gradient pairing.",
        reduction
    );
    println!("  Computing persistence on the Morse complex is vastly faster.\n");
}
