# Experiment conditions and results

## Ecole baseline

Measurements: `results/ecole/baseline.json`. Generate the results table with `python results/summarize_baseline.py`.

| Condition | Setting |
|---|---|
| Evaluation set | 400 records: 100 each of set covering, combinatorial auction, facility location and independent set |
| Instance identifiers | Seeds 301–400 within each family |
| Compared methods | LLM selection, EPM top-1, ISAC+, ISAC-SMAC, ISAC-ROAR; one shared default |
| Nomination width | 10 candidates in all 400 v3 selection records |
| Timing repetitions | Three stored timings per arm and instance; median used |
| Solver time limit | 10 seconds in the final measurement metadata |
| Recorded measurement date | 2026-08-13; the timestamp has no recorded timezone |
| Selection source | Main-table metadata points to `FINAL_llm_v3.json`; one corresponding selection is included in `examples/selection.json` |
| Historical hardware / OS / exact software versions | Not fully recorded; no historical lock is claimed |

The measurement implementation uses round-robin arm timing and sets numerical-library thread limits to one. The exact historical software revision and hardware configuration are unavailable.

## Method configuration in the available source

The following settings describe the included implementation. The exact historical API and memory versions are unavailable.

| Component | Supplied setting |
|---|---|
| EPM | XGBRegressor, 400 estimators, depth 7, learning rate 0.05, subsample and column subsample 0.8, min child weight 3, reg_alpha 0.1, reg_lambda 1.0, random_state 42 |
| Training inputs | 120 cached instances; 50 measured configurations per instance; 20 instance features plus encoded configuration |
| Training target | Log ratio of default runtime to candidate runtime |
| Candidate portfolio | The supplied 1,000-configuration pool; original order and candidate IDs preserved |
| LLM YAML | OpenAI-compatible provider; model `qwen3.7-max`, temperature 0.1, max_tokens 4096; exact historical API snapshot unknown |
| Memory | Current `skills/skill.md` retained as an executable example; final main-table memory identity unresolved |
| Runtime sample | One existing set-covering LP at seed 301; this is an interface example, not an additional validation dataset |

The compact training input keeps only fields read by the Ecole EPM: the first 20 features, default runtime, each configuration's values and runtime. Values are copied without recomputation. Redundant cache fields, intermediate responses and iteration records are omitted. Historical EPM weights were not found; future execution fits from these inputs.

## Metrics

The replay reports the stored score from the final timing records. For solver speedup, each instance uses default median runtime divided by the method median runtime, with the latter floored at 0.001 seconds. SGM is `exp(mean(log(1 + speedup))) - 1`, with a dimensionless shift of one.

SGM aggregates per-instance speedups. It is not the ratio of total runtimes. Neither SGM nor the timing records include the LLM call cost. The default row uses the historical baseline convention score=0 and speedup=1.

| Method | Mean stored score | SGM |
|---|---:|---:|
| Default | 0.000000 | 1.000000 |
| LLM | 0.563817 | 4.481359 |
| EPM top-1 | 0.542901 | 4.280673 |
| ISAC+ | 0.544122 | 4.294848 |
| ISAC-SMAC | 0.493336 | 3.660222 |
| ISAC-ROAR | 0.431754 | 3.042050 |

The table summarizes the stored final measurements.

## Additional results and data availability

The multi-family summary JSON files are retained under `results/portfolio-study/` for transparency. They contain previously reported aggregates, including k=10, but the corresponding k=10 per-instance raw records have not been located. The available current raw file is k=15, and is deliberately not substituted or included as if it reproduced k=10. This repository therefore does not claim that those summaries have been independently reproduced.

Selected MIPLIB final result records are under `results/miplib/`. External instance binaries and their exact version/source mappings are not bundled. They support inspection of recorded results, not complete replay of that experiment.

Exact reproduction is limited by the unavailable historical prompt/memory versions, EPM weights, API/software/hardware details and k=10 raw data.
