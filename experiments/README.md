# `experiments/` — hyperparameter sweep archive

This directory holds frozen snapshots of past runs. Each subdirectory is one
hyperparameter configuration; inside it you get the full `results/`, `runs/`
output and the exact `config.py` used.

Snapshots are created by [`src/data_compiler.py`](../src/data_compiler.py):

```bash
python -m src.data_compiler
```

It copies `results/`, `runs/`, and `src/config.py` into a new folder whose name
encodes 17 config values from `config.py`.

## Folder-name schema

```
<BUDGET>_<PRIOR_MAX_PLY>_<OPENING_REPLACEMENT_MAX_PLY>_
<POP_SIZE_REPERTOIRES>_<POP_SIZE_OPPONENTS>_<N_GENERATIONS>_
<TOURNAMENT_SIZE>_<CROSSOVER_RATE>_<MUTATION_RATE>_
<MUTATION_RETRIES>_<HOF_SIZE>_<OPPONENT_MUTATION_STRENGTH>_
<NOVELTY_WEIGHT>_<OPPONENT_CROSSOVER_RATE>_<OPPONENT_MUTATION_RATE>_
<LAMBDA_WEIGHT>_<MAIN_LAMBDA>
```

Field-by-field decoder:

| # | Field | Section in [config.py](../src/config.py) |
|---|---|---|
| 1  | `BUDGET`                       | Repertoire chromosome |
| 2  | `PRIOR_MAX_PLY`                | Scoring |
| 3  | `OPENING_REPLACEMENT_MAX_PLY`  | Repertoire chromosome |
| 4  | `POP_SIZE_REPERTOIRES`         | Genetic algorithm |
| 5  | `POP_SIZE_OPPONENTS`           | Genetic algorithm |
| 6  | `N_GENERATIONS`                | Genetic algorithm |
| 7  | `TOURNAMENT_SIZE`              | Genetic algorithm |
| 8  | `CROSSOVER_RATE`               | Genetic algorithm |
| 9  | `MUTATION_RATE`                | Genetic algorithm |
| 10 | `MUTATION_RETRIES`             | Repertoire chromosome |
| 11 | `HOF_SIZE`                     | Genetic algorithm |
| 12 | `OPPONENT_MUTATION_STRENGTH`   | Opponent evolution |
| 13 | `NOVELTY_WEIGHT`               | Opponent evolution |
| 14 | `OPPONENT_CROSSOVER_RATE`      | Opponent evolution |
| 15 | `OPPONENT_MUTATION_RATE`       | Opponent evolution |
| 16 | `LAMBDA_WEIGHT`                | Fitness function |
| 17 | `MAIN_LAMBDA`                  | Experiment matrix |

Example: `25_10_5_50_50_100_2_1.0_0.5_5_10_0.5_0.5_1.0_0.5_1.0_1.0` is
`BUDGET=25`, `MAX_PLY=10`, … `MUTATION_RATE=0.5`, …, `LAMBDA_WEIGHT=1.0`.

## Special folders

- **`FINAL - 10 Budget/`** and **`FINAL - 25 Budget/`** — the runs that
  back the figures and tables in `paper/results_10/` and `paper/results_25/`.
- **`old/`** — earlier iterations kept for reference; not used in the paper.

## Re-running one snapshot

```bash
# 1. Restore that folder's config.py
cp experiments/<folder>/config.py src/config.py

# 2. Wipe live results / runs (see ../src/clear.py for what gets removed)
python -m src.clear --all-files

# 3. Re-run
python src/experiments.py && python src/analyze.py
```
