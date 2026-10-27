# NEEDS — topo-sonify

Working ledger. Read on entry, update on exit.

## Goals
- Topology-first sonification: simplicial complexes, invariants, instruments, audio (Rust core, WASM and Web Audio front end).
- Keep tests verifying mathematical correctness (Betti numbers of known spaces).
- Direction under consideration (proposal, not a plan): explore how sound exists and propagates on strange spaces.
  The wave equation u'' = -L u on the Hodge Laplacian gives resonant frequencies sqrt(lambda). Candidate spaces:
  fractal graphs, percolation clusters, trees and mazes, quasicrystals, cell complexes, disordered media.
  Each candidate has a published value to test against before any sound is made.

## Status
- Research prototype, April to July 2026. Math core, instruments and WASM build work; 44 tests pass; clippy is clean.
- Development paused after 2026-07-13. README, LICENSE and this ledger were brought up to date on 2026-10-27.

## Blockers
- **GitHub Pages is not enabled for this repository.** The Pages workflow has failed since 2026-06-15 with
  "Failed to create deployment (404) ... Ensure GitHub Pages has been enabled". The workflow itself is fine.
  Fix: repository Settings, Pages, Source: GitHub Actions (or `gh api -X POST repos/marchdown/topo-sonify/pages -f build_type=workflow`).
  Until then the docs and lessons are readable only as files in the repository.

## Needs
- Owner decisions: enable Pages; whether the exploration direction above is a new phase of this repository.
- If it is: first milestone is modal synthesis plus an offline WAV writer, checked against the exact spectra of the
  path and cycle graphs (lambda_k = 2 - 2cos(k pi / n), 2 - 2cos(2 pi k / n)). No new dependencies needed.
  Next: Sierpinski gasket spectrum and its spectral dimension 2 ln 3 / ln 5, then bond percolation (p_c = 1/2).

## Notes
- Ledger established 2026-06-03.
- Not re-tested on 2026-10-27: browser audio playback of `web/index.html`. Tests, example, and WASM build were run.

_updated: 2026-10-27 · session: Claude Code, branch polish/readme-license-ledger_
