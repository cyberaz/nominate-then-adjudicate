# Contributing

Describe the affected method or experiment, the expected behavior, and how to run the changed code. Keep changes to input construction, configuration selection, solver behavior and metrics explicit.

Run the lightweight examples before submitting a change:

```bash
python examples/replay_selection.py
python results/summarize_baseline.py
```

Use a new output path for new experiments and retain the published measurement files. Document the environment, model, memory, portfolio and random seeds for new results. Do not commit API credentials or machine-specific configuration.

Original code contributions use the project's MIT license. Preserve applicable attribution and permissions when adding third-party code or data.
