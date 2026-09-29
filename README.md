# Omega Chess960 Selfplay

Self-play pipeline for **Omega `main`** using the current NimasBot Chess960 Polyglot book as the opening seed.

## Goal

1. Download the current NimasBot seed book: `books/c860/c960_280825_0508.bin`.
2. Clone and build the latest private `FayE75/Omega` `main` branch.
3. Start each self-play game from a Chess960 position covered by the seed book.
4. Follow weighted-random seed-book moves for the configured opening depth.
5. Continue with Omega vs Omega.
6. Save all games to PGN.
7. Reinforce winning/drawing opening moves and add new Omega continuations.
8. Build a full `c960_enriched.bin` while keeping the original book as the immutable base.

The repository stores only the compact cumulative learning map. The large enriched `.bin` and PGN are GitHub Actions artifacts, avoiding 80+ MB binary commits on every run.

## One-time setup

Create a repository secret named **`OMEGA_PAT`** with read access to `FayE75/Omega`. This is the same idea as the secret currently used by NimasBot to clone Omega.

## Run

Open **Actions → Omega Chess960 Selfplay → Run workflow**.

Recommended first smoke test:

- games: `10`
- nodes: `5000`
- book_plies: `12`
- train_plies: `40`

After that, a practical CPU-run starting point is:

- games: `200`
- nodes: `30000`
- book_plies: `12`
- train_plies: `40`

For higher-quality but slower learning, increase nodes rather than engine threads. Each game uses two Omega processes with one thread each.

## Outputs

Every workflow run uploads:

- `selfplay.pgn` — complete Chess960 self-play games.
- `c960_run_delta.bin` — only moves learned in the current run.
- `c960_enriched.bin` — original NimasBot seed book plus all cumulative learned moves/weights.
- `run_updates.json.gz` — compact current-run learning data.
- `cumulative_updates.json.gz` — cumulative learning data.
- `summary.json` — result counts, Omega SHA, book SHA, runtime, and learning statistics.

## Learning rule

For opening moves within `train_plies`:

- win from mover's perspective: `+8`
- draw: `+3`
- loss: `+0`

Weights are capped at the Polyglot maximum (`65535`). The existing seed-book weights are preserved and cumulative learning is added on top.

## Why the seed book is not overwritten

The seed book remains the trusted base. Self-play produces a sparse learning layer. On every run, the workflow reconstructs the enriched book as:

`current NimasBot seed book + cumulative Omega self-play learning`

This makes the process reversible and avoids damaging the original book if a self-play batch is poor.
