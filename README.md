# topo-sonify

Turns simplicial complexes into sound. The topology comes first: complexes, homology, persistent homology,
Hodge Laplacians and discrete Morse theory are built and tested against known results, and the instruments
map those invariants to audio afterwards.

**Status:** research prototype, April to July 2026. The math core and a WASM build work. Development is
paused; `NEEDS.md` has the current state and the next steps.

## What is here

- `src/topology/`: simplicial complexes, Vietoris-Rips from 2D points, Z/2 homology and Betti numbers,
  persistent homology (column reduction with the twist optimisation), the Hodge Laplacian over the reals with
  its eigenvalues, discrete Morse pairing, bottleneck distance.
- `src/audio/`: oscillators, biquad filters, delay, reverb, distortion.
- `src/instruments/`: `BettiDrone`, `SimplicialPluck`, `MorseSynth`.
- `src/mapping/`: `TopologySonifier` connects the topology to the instruments. `src/lib.rs` is the
  wasm-bindgen bridge, and `web/index.html` is the browser front end.
- `docs/`: a five-chapter seminar on the mathematics, starting at `docs/index.html`.
- `lessons/`: six interactive lessons (points, connections, loops, persistence, curvature, playground).

## Run it

```sh
cargo test                                    # 49 tests
cargo run --example ch1_simplicial_complexes  # also ch2_homology, ch3_persistence
wasm-pack build --target web --release        # about 20 s, about 158 KB of wasm
```

For the browser version, serve the repository root with any static file server after the WASM build and open
`/web/`.

## How it is checked

The tests assert known facts before anything is audible. The hollow tetrahedron (a 2-sphere) has Betti numbers
(1, 0, 1) and Euler characteristic 2. The kernel of the Hodge Laplacian has the same dimension as the Betti
numbers. The Morse pairing satisfies the Morse inequalities and recovers the Euler characteristic from its
critical cells. The two persistence algorithms agree, and the bottleneck distance of a diagram to itself is zero.
`METHODOLOGY.md` records the order the project was built in, and `DEVNOTES.md` records the bugs that were found
and fixed. `docs/LITERATURE-GROUNDING.md` lists the references.

## License

MIT, see `LICENSE`.
