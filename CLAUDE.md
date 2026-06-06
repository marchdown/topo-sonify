# topo-sonify

Topology-first sonification engine. Simplicial complexes become sound.

## Architecture

```
Topology (Rust)  →  Invariants  →  Instruments  →  Effects  →  Output (WASM/Web Audio)
```

- `src/topology/` — mathematical core: simplicial complexes, homology, persistent homology
- `src/audio/` — oscillators, biquad filters, effects (delay, reverb, distortion)
- `src/instruments/` — BettiDrone, SimplicialPluck, MorseSynth
- `src/mapping/` — TopologySonifier: wires topology → audio
- `src/lib.rs` — WASM bridge via wasm-bindgen
- `web/` — browser frontend (Web Audio API, Canvas)
- `docs/` — literate HTML documentation with MathJax

## Key mappings

| Invariant | Audio Parameter | Module |
|-----------|----------------|--------|
| β₀ (components) | oscillator count | BettiDrone |
| β₁ (loops) | LFO modulation | BettiDrone |
| β₂ (voids) | sub-oscillators | BettiDrone |
| f-vector | just intonation ratios | BettiDrone |
| χ (Euler char) | reverb character | CurvatureReverb |
| persistence pairs | delay taps | PersistenceDelay |
| filtration sweep | musical score | MorseSynth |
| Betti numbers | wavefold stages | BettiDistortion |

## Build

```sh
cargo test                    # run all tests
cargo build --release         # native build
wasm-pack build --target web  # WASM build (requires wasm-pack)
```

## Conventions

- Rust only. No Python, no Node.
- All topology code over Z/2Z (mod 2 coefficients).
- Tests verify mathematical correctness (Betti numbers of known spaces).
- Literate docs: math first, code second.

## Working ledger
- This strand keeps a local `NEEDS.md` — goals, blockers, and needs — read on entry, updated on exit. Convention: `~/.claude/CLAUDE.md` → "Working ledger: NEEDS.md per strand".
