# Decision log (working notes)

Decisions agreed so far. This file feeds `docs/design.md`, which will be written once the whole architecture and the implementation plan are agreed. It is not a build specification yet.

## Project

- **Goal:** answer three questions with simulation, exact methods and RL:
  - **Q1:** fastest single-player strategy,
  - **Q2:** how much strategy matters vs luck in two-player games,
  - **Q3:** how a rival changes the optimal strategy.
- **Deliverables:**
  - a simulation package,
  - experiment scripts,
  - a LaTeX report,
  - a public GitHub repo readable by recruiters.
- **Players:** 2–6, plus single-player mode.
- **Rules:** `docs/rules.md` is authoritative.

## Board data

- **Files:** `data/board/nodes.csv` (`id,name,kind,special,x,y`) and `data/board/edges.csv` (`from,to,mode,dots`). Transcribed by hand from the physical board.
- **Conventions:**
  - `dots` is the number of dots strictly between the endpoints.
  - `mode` is `land`, `sea` or `air`; air edges have no dots.
  - `kind` is `start`, `red`, `junction` or `dot`; `dot` is used only for the Sahara trap.
  - `x,y` are pixel coordinates in `docs/context/afrikan-tahti-board.jxl` (376×512, origin at top-left, y pointing down).
- **Checks:**
  - Verified edge-for-edge against tjrantal/AfrikanTahti (GPL-3.0). Credit this source.
  - Expected totals: 2 start cities, 30 red circles, 111 land locations, 77 sea locations, 220 locations in total.
  - Etzell's thesis reports 112 land locations (221 in total). Our transcription of the physical board and the Java data agree on 111. Mention this in the report.

## Compute

- Rust engine on CPU; no JAX or GPU engine.
- Rent a GPU only if network training turns out to be the bottleneck.

## A1. Repo structure and tooling

- A single maturin "mixed" project:
  - Rust crate in `src/`
  - Python package in `python/afrikan_tahti/`
  - `data/board/`, `scripts/`, `tests/`, `report/`, `docs/`
- **Tooling:** uv for the Python environment (`uv.lock` committed; uv version pinned in the README); maturin ≥ 1.15 as the build backend.

## A2. The rules live only in Rust

- Python never reimplements the rules.
- The board CSVs are parsed in Rust by our own small parser (no `csv` crate).
- The default board is embedded at compile time.
- **Validation at load time** fails fast with a clear message:
  - location counts as listed under Board data,
  - connectivity,
  - valid ids,
  - no junction mixes land and sea.

## A3. Public and hidden state

- **Public state:**
  - per player: position, money, flags,
  - opened circles,
  - remaining disk counts,
  - Cape Town claimed,
  - Star found,
  - turn and phase.
- **Hidden state:**
  - the disk layout, shuffled at setup as in the real game,
  - the RNG streams for the dice.
- Search and exact solvers work only on the public state, where opening a disk is a chance event weighted by the remaining counts.
- A fixed layout and fixed dice streams make two things possible:
  - duplicate (common-random-numbers) evaluation,
  - an oracle player for the skill measure.

## A4. Our own random number generator

- **Algorithm:** xoshiro256++, seeded through SplitMix64.
- **Streams:** separate streams for the layout and for each seat's dice. Parallel workers get independent streams, derived from the main seed and the stream id.
- **Range sampling:** unbiased, using rejection (Lemire's method). Shuffling with Fisher–Yates.
- **Required tests:**
  - known-answer tests against the reference outputs of xoshiro256++ and SplitMix64,
  - chi-square uniformity tests for die rolls and shuffle positions,
  - determinism: the same seed gives the same game.

## A5. The action model

- **Before the roll**, choose one of:
  - pay to open,
  - roll to open,
  - roll,
  - free sail,
  - fly without rolling (only when held at St Helena).
- **After the roll** (or after choosing free sail), choose one of:
  - walk or sail to a destination, then open or don't open,
  - fly to a destination, then open or don't open,
  - stay put.
- **Path choice:** for each destination the engine picks the path that leaves the most money (sea fares, Cape Town bonus). Paths to the same destination differ only in money, so nothing is lost.
- **Encoding:** a fixed-size action space (about 500 actions) with a legal-action mask.
- **Turn cap:** after 1000 rounds the game ends with no winner. The number of such games is reported.

## A6. Python interface

- **`Game`:** a single game, stepped one action at a time; it exposes the public state.
- **`VecEnv`:** N parallel games.
  - Observations and masks are returned as NumPy arrays; actions are passed in as an array.
  - Opponents are played by built-in Rust policies or by Python.
- **`simulate(policies, n_games, players, seed)`:** whole games run in Rust, in parallel with `std::thread::scope`.
- **Baseline policies in Rust:** Random, Close, Frugal and Fly (as in Etzell), behind a common `Policy` interface.
- **Rust dependencies:** `pyo3` ≥ 0.29 (fixes RUSTSEC-2026-0176 and -0177) and `numpy` (rust-numpy) 0.29. Nothing else.

## Out of scope

- A GUI or human play.
- A Gymnasium/PettingZoo dependency (we follow their conventions only).
- Etzell's TSP and Dominator strategies.
- JAX.

## Still to decide

- **Baselines and search:** exactly which baselines, and whether to include an expectimax or MCTS player.
- **Exact methods:** the return-trip dynamic program, and reduced boards.
- **RL:** algorithm, library or own implementation, observation encoding, reward, self-play setup.
- **Experiments:** design and statistics (duplicate games, confidence intervals, skill measure).
- **Report:** toolchain and figures.
- **Repo setup:** CI, licence.
- **Implementation plan:** the steps and their acceptance criteria.
