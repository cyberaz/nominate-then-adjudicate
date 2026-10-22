# Data

| Path | Contents |
|---|---|
| `results/ecole/baseline.json` | Final measurements for 400 instances: one shared default and five methods, with three timings per instance and arm |
| `examples/selection.json` | One recorded top-10 shortlist, chosen candidate and configuration |
| `examples/training_input.json` | Twenty instance features and measured configuration runtimes for 120 training instances |
| `experiments/ecole-miplib/data/results/instances/` | One set-covering LP for the single-instance example |
| `data/config-pools/` | Candidate configuration portfolios |
| `experiments/*/skills/skill.md` | Memory examples for each implementation |
| `results/portfolio-study/` | Multi-family aggregate results |
| `results/miplib/` | Selected MIPLIB results |

The training input contains only the fields consumed by the EPM: instance features, default runtimes, configuration values and configuration runtimes. Family names, seeds, candidate IDs and variable/constraint names retain their experimental meaning.

The complete evaluation instances, historical memory versions and model weights are not included. The multi-family k=10 aggregates lack their per-instance raw records, and MIPLIB instance binaries are not distributed here. See [EXPERIMENT.md](EXPERIMENT.md) for the resulting reproduction limits.

## Licensing

The MIT license covers original project code and authored documentation. It does not automatically license datasets, model outputs, upstream source or dependencies. No separate data license is specified for the included experimental materials. PySCIPOpt/SCIP, XGBoost and other dependencies retain their own licenses.
