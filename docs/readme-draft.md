# Afrikan tähti: optimal strategy

*Afrikan tähti* (The Star of Africa, 1951) is a classic Finnish board game. Players travel around Africa by land, sea and air, opening hidden disks to find the African Star and bring it back to Tangier or Cairo. The game is famously luck-driven. This project asks how much good strategy can still change the outcome.

## Research questions

1. **Single player:** what strategy finds the Star and brings it home in the fewest turns?
2. **Two players:** what is the best strategy, and how much better is it than a baseline? How much of the game is skill and how much is luck?
3. **Interaction:** how much does the optimal strategy change when a rival is present?

## Approach (planned)

- A fast **game engine in Rust**, used from Python through PyO3 bindings (maturin, uv).
- **Exact dynamic programming** where the state space allows it, e.g. the return trip with the Star.
- **Reinforcement learning** (Python/PyTorch) for the full game. Its results are checked against the exact solutions.
- **Large-scale simulation** with paired "duplicate" games and confidence intervals, to measure how much the outcome depends on skill and how much on luck.
- The results are collected in a **LaTeX report**.

## Status

In design. The rules are fixed and the board has been transcribed and validated; the engine is next.

## Repository

| Path | Contents |
|---|---|
| `docs/rules.md` | The rules as implemented: the 2005 official rules plus clarifications for ambiguous cases |
| `data/board/` | The board as a graph: `nodes.csv` and `edges.csv`, transcribed by hand from the physical board |
| `docs/context/` | Background material and study notes |

## Board data

The board is represented as a graph:
- **Vertices:** the 2 start cities, the 30 red circles, the road and sea junctions, and the Sahara trap.
- **Edges:** land and sea routes with dot counts, and air routes.

The engine expands this into the 220 individual locations the pieces move on.

The transcription was checked edge by edge against the board in Daniel Suni's and Timo Rantalainen's Java implementation ([tjrantal/AfrikanTahti](https://github.com/tjrantal/AfrikanTahti), GPL-3.0), and against the location counts reported by Etzell (2025).

## Related work

- P. Etzell, *On the gameplay of Afrikan tähti*, Master's thesis, Aalto University, 2025. It uses expectiminimax search against simple strategies and proves that a central decision problem of the game is NP-hard.

## Disclaimer

Afrikan tähti was designed by Kari Mannerla and is published by Peliko/Martinex. This is an independent hobby and research project, not affiliated with the designer or the publisher.

## License

TBD
