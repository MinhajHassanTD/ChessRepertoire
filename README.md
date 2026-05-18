# Co-Evolutionary Genetic Algorithms for Chess Opening Repertoires

[![tests](https://github.com/MinhajHassanTD/ChessRepertoire/actions/workflows/tests.yml/badge.svg)](https://github.com/MinhajHassanTD/ChessRepertoire/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Full title:** Co-Evolutionary Genetic Algorithms for Memorization-Constrained Robust Chess Opening Repertoires

Minhaj ul Hassan, Hazeera Muhammad Hashim, Zainab Zahid
Department of Computer Science, Habib University

> Final project for *Computational Intelligence* (Spring 2026). Paper is in [paper/main.tex](paper/main.tex).

---

## TL;DR

We frame chess opening preparation as a **budget- and closure-constrained combinatorial optimization** problem over a real Lichess position graph, and compare five methods over 30 seeds: most-played heuristic, random search, greedy hillclimber, a standard GA (`STATIC`), and a co-evolutionary GA (`COEVOLVE`).

**Headline findings:**

| Finding | Evidence |
|---|---|
| The **closure rule** is the dominant factor in held-out performance | $+0.023$ score, $A_{12} = 1.0$, $p < 7.5 \times 10^{-9}$ vs. no-closure |
| **`COEVOLVE` does not beat `STATIC`** despite a $\sim 35\times$ compute cost | $A_{12} = 0.45$, $p = 1.0$ |
| **Greedy hillclimber matches GA-quality solutions** at a fraction of runtime | landscape responds well to local search |

The negative result on co-evolution is reported honestly; the positive result is the closure-constrained chromosome representation.

---

## What the project does

A chess **opening repertoire** is a pre-planned set of moves: when White plays `1.e4`, Black might respond `1...e5`, then `2.Nf3`, etc. The branching sequences form a tree.

We ask: *given a strict budget of committed moves per color, what opening repertoire performs best across players of different skill levels?*

Two ideas drive the design:

1. **Closure-constrained chromosome.** The repertoire tracks not just *our* moves but all common opponent replies that must be covered. Any opponent move played in $\geq 15\%$ of real games at a reached position is automatically added to the coverage set. This mirrors how a real player thinks: "I play 1.e4, so I must have answers to 1...e5, 1...c5, 1...e6, and 1...c6."

2. **Co-evolutionary opponent.** An adversarial population of opponents (mixtures over three skill bands) evolves *alongside* the repertoires, probing for weaknesses to prevent overfitting.

Fitness combines mean held-out score with **CVaR** (worst single rating band):
$$\text{fitness} = \overline{\text{score}} + \lambda \cdot \text{CVaR}, \qquad \lambda = 1.0$$

---

## Methods compared

| Method | Description | Evolves? |
|---|---|---|
| `MOST_PLAYED` | Always pick the most-played move. No optimization. | No |
| `RANDOM_SEARCH` | Sample random repertoires, keep the best. | No |
| `GREEDY_HILLCLIMB` | Start greedy, accept any mutation that improves fitness. | No |
| `STATIC` | GA with a fixed uniform-mixture opponent | GA only |
| `COEVOLVE` | Both repertoire and opponent populations evolve together | Yes |

Ablations: `STATIC_NOCLOSURE` / `COEVOLVE_NOCLOSURE` (does closure help?), closure-threshold sweep, budget sweep.

---

## Project structure

```
ChessRepertoire/
├── README.md                ← this file
├── requirements.txt
├── .env.example             ← template for Lichess API token
│
├── src/                     ← all code (components C1..C10)
│   ├── config.py            ← all tunable constants
│   ├── data_ingest.py       ← C1: Lichess Opening Explorer crawler
│   ├── graph.py             ← C2: position graph builder
│   ├── policies.py          ← C3: per-band move distributions
│   ├── eval_cache.py        ← C4: Bayesian-shrunk position scores
│   ├── repertoire.py        ← C5: closure-constrained chromosome + operators
│   ├── fitness.py           ← C6: walk + fitness
│   ├── opponent.py          ← C7: opponent chromosome
│   ├── coevolution.py       ← C8: main GA loop
│   ├── experiments.py       ← C9: experiment matrix dispatch
│   ├── analyze.py           ← C10: figures, tables, stats
│   ├── data_compiler.py     ← snapshots results/ + runs/ + config.py into experiments/
│   └── clear.py             ← wipes *.pkl files in runs/ so experiments can be rerun
│
├── data/                    ← databases + pickled graphs / caches (gitignored)
├── runs/                    ← per-run result pickles (gitignored)
├── results/                 ← figures, tables, repertoire dumps
├── experiments/             ← frozen hyperparameter-sweep snapshots ([README](experiments/README.md))
├── visualize/               ← standalone HTML visualizers (Sankey, 3D graph)
├── tests/                   ← pytest suite, one file per component
└── paper/                   ← IEEE-conference LaTeX source + final figures
    ├── main.tex
    ├── references.bib
    ├── results_10/          ← all figures/tables for the budget-10 run
    └── results_25/          ← all figures/tables for the budget-25 run
```

---

## Setup

**Requirements:** Python 3.10+, a free [Lichess](https://lichess.org) account for the API token.

```bash
pip install -r requirements.txt
cp .env.example .env
# Paste your token from https://lichess.org/account/oauth/token into .env
```

The `.env` file is gitignored and must never be committed.

---

## Running the pipeline

Each step depends on the previous. The data pipeline (C1–C4) only needs to run once.

```bash
# C1 — crawl Lichess Opening Explorer (resumable; skip if data/snapshot.db exists)
python src/data_ingest.py

# C2 — build position graphs (train + held-out)
python src/graph.py data/snapshot.db

# C3 — per-band move policies (also prints a band-separation diagnostic)
python src/policies.py

# C4 — Bayesian-shrunk eval cache
python src/eval_cache.py

# C9 — main experiments (resumable; skips already-completed runs)
python src/experiments.py

# Optional ablations: closure on/off, threshold sweep, budget sweep
python -m src.experiments ablations

# C10 — figures, tables, statistical tests
python src/analyze.py             # core outputs
python src/analyze.py --appendix  # + appendix figures

# Tests
pytest tests/

# Wipe runs/*.pkl so experiments rerun from scratch
python -m src.clear            # deletes *.pkl in runs/
python -m src.clear --all-files  # delete every file in runs/
```

---

## Key parameters (defaults)

| Parameter | Value |
|---|---|
| Max ply depth | 10 |
| Budget per color | 25 (main); also 10 in [paper/results_10/](paper/results_10/) |
| Closure threshold | 15% |
| Rating bands | 1000–1399, 1400–1799, 1800–2199 |
| Fitness $\lambda$ (CVaR weight) | 1.0 |
| Population — repertoires / opponents | 50 / 30 |
| Generations | 100 |
| Seeds | 30 (`1000`–`1029`) |
| Train/held-out cutoff | 2025-06 |

All knobs live in [src/config.py](src/config.py).

---

## Outputs and how to read them

Generated under `results/` (and copied per-budget into `paper/results_10/`, `paper/results_25/`):

| Artifact | What it shows |
|---|---|
| `summary_table.csv` | Mean / std / Wilcoxon / $A_{12}$ for every method |
| `pairwise_table.csv` | All method-vs-method Holm-corrected comparisons |
| `closure_ablation_table.csv` | The headline closure result |
| `convergence.png` | GA training curves with 95% CI |
| `score_distributions.png` | Box + strip plots of held-out scores |
| `band_breakdown.png` | Per-rating-band scores |
| `closure_ablation.png` | Closure ON vs OFF |
| `ga_vs_nonga.png` | GA methods against greedy / random baselines |
| `repertoire_structure.png` | Tree depth / branching statistics |
| `coevolve_dynamics.png` | Hall-of-Fame and opponent-population dynamics |
| `repertoire_graph_*.png` / `repertoire_sankey_*.png` | Best-repertoire visualizations per method |
| `repertoire_tree.txt`, `repertoire_grouped.txt` | Human-readable best repertoires |
| `runtime_table.csv` | Median wall time per method |

**Held-out evaluation always uses a uniform opponent mixture** regardless of training mode, so cross-method comparisons are fair.

**Fitness $> 1.0$ is correct.** With $\lambda = 1.0$ the theoretical max is $2.0$ (mean $+$ CVaR, each $\in [0,1]$). Values around $1.0$ mean roughly random-play level on both objectives simultaneously.

---

## Reproducibility

A single `np.random.default_rng(seed)` is created per run and threaded through every component — no other random source is used. Given the same seed, runs are **bit-identical**. The result dict for each run includes the seed, full config, and the current git commit.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute with attribution.

## Citation

If you build on this work, please cite the paper (see [paper/main.tex](paper/main.tex)). BibTeX entry will be added once the paper is finalized.
