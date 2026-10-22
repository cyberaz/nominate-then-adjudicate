# Nominate-then-Adjudicate

Research implementation of **Nominate-then-Adjudicate: LLM-Assisted One-Pass and Per-Instance MIP Solver Tuning**, accepted as a short paper at IEEE BigData 2026.

The method uses an empirical performance model to nominate a small set of SCIP configurations, then asks a language model to select one using instance structure and a textual skill memory.

This repository contains the method code, a runnable example, the inputs needed by that example, and final baseline measurements. Experiment conditions, reported results and data availability are documented in [EXPERIMENT.md](EXPERIMENT.md). Iteration histories, old checkpoints and bulk instance archives are intentionally excluded.

## Run the offline example

Python 3.11, standard library only:

```bash
python examples/replay_selection.py
python results/summarize_baseline.py
```

The example replays one saved selection and its corresponding final measurement. The summary script calculates the 400-instance baseline table. Neither command trains a model, contacts an API or runs SCIP. Both use the included experimental records.

## Install the method implementation

```bash
python -m pip install -e '.[llm]'
python code/scripts/run.py --list
python code/scripts/run.py ecole-miplib epm_with_selection.py --help
```

The package requires Python 3.11 or newer. The main dependencies are PySCIPOpt, XGBoost, NumPy, SciPy and scikit-learn; the `llm` extra adds the API clients. SMAC baselines additionally require the `smac` extra. Dependency ranges are in `pyproject.toml`; they are not an assertion of the historical environment.

## Run the one-instance method example

The repository includes a compact training input (120 instances and 6,000 measured configuration pairs), a 1,000-configuration portfolio, one LP instance and a current memory example. Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` in your process environment, and set an accessible model in `experiments/ecole-miplib/config/experiment.yaml`.

```bash
python code/scripts/run.py ecole-miplib epm_with_selection.py \
  --cache ../../examples/training_input.json \
  --n-eval 1 \
  --top-k 10 \
  --skill skills/skill.md \
  --output data/runs/example.json
```

This command fits the EPM, calls the configured API and evaluates configurations with SCIP. It can incur API charges. The current memory and API configuration are not established as the exact historical main-table versions. Only the first evaluation LP is included; do not remove `--n-eval 1` expecting the complete historical benchmark to be available.

The multi-family implementation is also included for method inspection. Its full training/evaluation inputs are not bundled; the missing k=10 historical records are explicitly documented. It is not covered by the one-instance runnable example.

## Results

Offline recomputation of the final matched baseline measurements:

| Method | Mean stored score | SGM solver speedup |
|---|---:|---:|
| Default | 0.0000 | 1.00 |
| Nominate-then-Adjudicate | 0.5638 | 4.48 |
| EPM top-1 | 0.5429 | 4.28 |
| ISAC+ | 0.5441 | 4.29 |
| ISAC-SMAC | 0.4933 | 3.66 |
| ISAC-ROAR | 0.4318 | 3.04 |

These statistics cover 400 instances and exclude LLM latency. See [EXPERIMENT.md](EXPERIMENT.md) for timing, metrics and known limits. The three timing repetitions are measurements of each instance, not three independent datasets.

## Layout

```text
code/src/al_param_tuning/       Shared method implementation
code/scripts/run.py            Experiment-aware entry point
experiments/ecole-miplib/      Main method, baselines, configuration and one LP
experiments/portfolio-study/   Multi-family method and its solver override
examples/                     Offline replay, one saved selection, training input
results/                      Final measurements, summaries and table generation
data/config-pools/            Required candidate portfolios
```

See [DATA_CARD.md](DATA_CARD.md) for the included data and its scope.

## License and citation

Original code is licensed under [MIT](LICENSE). Data and third-party material are covered separately in [DATA_CARD.md](DATA_CARD.md); no blanket data license is asserted. See [CITATION.md](CITATION.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

