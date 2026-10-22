"""
Skill Manager (Multi-Parameter Version)

Manages the skill.md file that accumulates parameter tuning knowledge
from random configuration experiments. Used by LLM to make better recommendations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..features.instance_descriptor import InstanceDescriptor


class SkillManager:
    """
    Manages the skill.md file that accumulates parameter tuning experience
    from random configuration experiments.
    """

    def __init__(self, skill_path: str, llm_client=None, max_length: int = 15000):
        """
        Args:
            skill_path: Path to skill.md file
            llm_client: Optional LLMClient instance for summarization
            max_length: Maximum length of skill file before compression
        """
        self.skill_path = Path(skill_path)
        self.skill_path.parent.mkdir(parents=True, exist_ok=True)
        self.llm_client = llm_client
        self.instance_descriptor = InstanceDescriptor()
        self.max_length = max_length

    def load(self) -> str:
        """Load skill file content."""
        if not self.skill_path.exists():
            return self._create_initial_skill()

        with open(self.skill_path, 'r', encoding='utf-8') as f:
            return f.read()

    def batch_update(self, experiment_results_list: List[Dict], max_concurrent: int = 8):
        """
        Batch update skill file with multiple experiment results in parallel.

        Args:
            experiment_results_list: List of result dicts
            max_concurrent: Maximum number of concurrent LLM calls (default: 8)
        """
        if not self.llm_client:
            for experiment_results in experiment_results_list:
                self.update(experiment_results)
            return

        # Build all prompts
        prompts = []
        for experiment_results in experiment_results_list:
            current_skill = self.load()
            raw_text = self._format_results([experiment_results])
            prompt = self._build_update_prompt(current_skill, raw_text)
            prompts.append(prompt)

        # Call LLM in parallel
        responses = self.llm_client.batch_ask(
            prompts=prompts,
            max_concurrent=max_concurrent,
        )

        # Append all responses to skill file
        with open(self.skill_path, 'a', encoding='utf-8') as f:
            for response in responses:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n\n## Update {timestamp}\n\n{response}\n")

    def update(self, experiment_results: Dict):
        """
        Update skill file with new random configuration experiment results.

        Args:
            experiment_results: Dict with keys:
                - features: list of 20 floats
                - random_configs: list of {config, result, exceeded}
                - default_result: {status, objective, time, gap}
                - best_config: dict
                - best_config_index: int
                - best_time: float
                - best_speedup: float
                - model: optional PySCIPOpt Model
        """
        current_skill = self.load()

        # Format raw results
        raw_text = self._format_results([experiment_results])

        # Build update prompt
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.llm_client:
            prompt = self._build_update_prompt(current_skill, raw_text)

            try:
                summary = self.llm_client.ask(prompt, max_tokens_override=1200)
                new_entry = f"\n\n## Update {timestamp}\n\n{summary}"
            except Exception as e:
                print(f"Warning: LLM summarization failed: {e}")
                new_entry = f"\n\n## Update {timestamp}\n\n(LLM summarization failed, raw data preserved)\n\n{raw_text}"
        else:
            new_entry = f"\n\n## Update {timestamp}\n\n{raw_text}"

        with open(self.skill_path, 'a', encoding='utf-8') as f:
            f.write(new_entry)

    def _build_update_prompt(self, current_skill: str, raw_text: str) -> str:
        """Build the prompt for skill update — analyzing all random configs (positive + negative)."""
        return f"""You are an expert in mixed-integer programming (MIP) and solver parameter tuning.

## Task
You are given detailed information about a MIP instance and the results of testing 20 random parameter configurations. Each random config independently samples a value for all 30 SCIP parameters uniformly from each parameter's candidate list (the default value is one of the candidates). Because candidate lists average ~4-5 values, on average only about 25% of parameters happen to retain their default in any given config — most parameters are changed. All configs were solved with an adaptive time limit of 1.5× the default solve time (floored at 0.5s, capped at 10s).

**Important**: You see ALL 20 configs — both fast (positive samples) and slow/timeout (negative samples). The negative samples are equally informative: they reveal which parameter values or combinations to AVOID.

You do NOT know what type of problem this is — you must infer it from the instance structure alone.

## Current Skill File
{current_skill}

## New Experiment Results
{raw_text}

## What to Analyze

### Step 1: Infer Problem Type and Extract Key Structural Features
The New Experiment Results above include BOTH a 20-dimensional feature vector AND the **raw
formulation** (full variable list, every constraint with its coefficients, and the objective).
The 20 features are **lossy aggregates** — they cannot capture symmetry, special/coupling constraint
patterns, or coefficient structure.

**You MUST read the raw constraint matrix directly, not only the 20 features.** From the raw
formulation infer:
- What type of MIP problem this likely is (from variable naming, constraint patterns like covering ≥, packing ≤1, assignment =1, flow conservation, big-M disjunctions)
- **Symmetry / special structure** — interchangeable variables/blocks, repeated coefficient rows, identical columns — since this strongly affects which parameters help
- Key structural characteristics (coefficient distribution, big-M, block structure) that affect parameter tuning

Ground your parameter insights in structure you actually see in the raw matrix, not just the aggregate features.

**CRITICAL**: You MUST output a "Key Structural Features" summary block:

```
#### Key Structural Features
- **Problem scale**: N vars (X binary, Y integer, Z continuous), M constraints
- **Matrix density**: D
- **Constraint structure**: E% equality, I% inequality
- **Inferred type**: TYPE — brief reason
```

### Step 2: Analyze Fast Configs (Positive Samples)
Identify the top 3-5 fastest configs and analyze what parameter values they share:
- Which specific parameter values appear most frequently in fast configs?
- Which parameters are most often changed from default in fast configs (i.e., matter most)?
- Are there parameter combinations that appear together in fast configs (suggesting positive interactions)?

Format as a comparison table:
| Config Rank | Time | Speedup | Key Parameter Values |
|-------------|------|---------|---------------------|
| #1 | 0.05s | 3.2x | maxnthreads=16, conflict=False, ... |
| #2 | 0.08s | 2.0x | maxnthreads=16, pricing=q, ... |

### Step 3: Analyze Slow/Timeout Configs (Negative Samples)
**CRITICAL**: Identify the slowest configs and those that timed out. Analyze what makes them bad:
- Which parameter values appear in slow configs but NOT in fast configs?
- Which parameter values or combinations consistently cause slowdowns?
- Are there "dealbreaker" parameters — values that almost always make things worse?

### Step 4: Extract Parameter Importance Ranking
Based on comparing fast vs slow configs, rank the 30 parameters by importance:
1. Which parameters, when changed, most consistently affect solve time?
2. For each important parameter, what values correlate with speedup vs slowdown?
3. Which parameters can safely stay at default?

### Step 5: Extract Generalizable Insights
Provide 2-3 deep, generalizable insights about parameter tuning for this problem structure:
- **Mechanistic understanding**: Explain WHY certain parameter values work better
- **Multi-parameter interactions**: Which parameter combinations amplify or cancel each other?
- **Generalizable patterns**: What would you recommend for SIMILAR problems?

## Output Format
Provide your analysis directly — it will be appended to the skill file as-is. Use markdown headings for the five steps. Keep the total length under 1000 words.

Your analysis:"""

    def _create_initial_skill(self) -> str:
        """Create initial skill file content."""
        content = """# SCIP Parameter Tuning Experience (Multi-Parameter Random Search)

This document accumulates parameter tuning experience from random configuration experiments.
Each experiment tests 20 random parameter configurations (30 params each independently sampled
uniformly from its candidate list, so on average only ~25% of params happen to keep their
default while most are changed) on MIP instances.

## General Principles
- Different problem families respond differently to parameter combinations
- Features like matrix density, variable types, and constraint structure guide tuning
- Some parameter values consistently correlate with faster solving across problems
- Negative samples (slow configs) are equally informative — they reveal what to avoid
- Multi-parameter interactions can produce effects not visible in single-parameter changes

## Accumulated Experience
(Will be updated as more instances are processed)
"""

        with open(self.skill_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return content

    def _format_results(self, results: List[Dict]) -> str:
        """
        Format experiment results for LLM summarization.

        Output format:
        1. Problem structure summary (one paragraph)
        2. All 20 random configs with full parameter values and solve times

        Args:
            results: List of experiment result dicts

        Returns:
            Formatted string for LLM prompt
        """
        lines = []

        for idx, result in enumerate(results):
            family_id = result.get('family_id', 'unknown')
            seed = result.get('seed', '?')
            lines.append(f"## Instance {idx + 1} ({family_id}, seed={seed})\n")

            # 1. Problem structure summary
            lines.append("### Problem Structure Summary\n")

            features = result.get('features', [])
            if features:
                n_constraints = int(features[0])  # row_count
                n_vars = int(features[1])          # col_count
                matrix_density = features[3]       # matrix_density
                n_binary = int(features[4] * features[1])      # binary_ratio * n_vars
                n_integer = int(features[5] * features[1])     # general_integer_ratio * n_vars
                n_continuous = int(features[6] * features[1])  # continuous_ratio * n_vars
                equality_ratio = features[8]       # equality_ratio
                inequality_ratio = features[9]     # inequality_ratio

                summary = (
                    f"This {family_id} instance has "
                    f"{n_vars} variables ({n_binary} binary, {n_integer} integer, {n_continuous} continuous) "
                    f"and {n_constraints} constraints. "
                    f"The constraint matrix has density {matrix_density:.3f} with "
                    f"{equality_ratio:.3f} equality constraints and {inequality_ratio:.3f} inequality constraints. "
                )

                if matrix_density < 0.1:
                    summary += "The problem is highly sparse. "
                elif matrix_density > 0.5:
                    summary += "The problem is relatively dense. "

                if n_binary / max(n_vars, 1) > 0.8:
                    summary += "Most variables are binary, suggesting a combinatorial optimization problem. "
                elif n_continuous / max(n_vars, 1) > 0.5:
                    summary += "Many continuous variables suggest LP-heavy solving. "

                lines.append(summary + "\n")
            else:
                lines.append("(No feature data available)\n")

            # 1b. Full raw planning formulation (complete constraint matrix)
            model = result.get('model')
            if model is not None:
                lines.append("### Full Planning Formulation\n")
                lines.append(self.instance_descriptor.describe(model) + "\n")

            # 2. Random configurations and results
            lines.append("### Random Configurations and Results\n")

            random_configs = result.get('random_configs', [])
            default_result = result.get('default_result', {})
            default_time = default_result.get('time', float('inf'))

            # Sort by solve time
            sorted_configs = sorted(
                enumerate(random_configs),
                key=lambda x: x[1].get('result', {}).get('time', float('inf'))
            )

            # Print default baseline
            lines.append(f"**Default Configuration**: time={default_time:.3f}s, "
                        f"status={default_result.get('status', 'unknown')}\n")

            # Print all 20 random configs
            lines.append("**Random Configurations** (sorted by solve time):\n")

            param_names = [
                'separating/clique/freq', 'misc/usesymmetry', 'lp/pricing', 'lp/refactorinterval', 'lp/scaling',
                'presolving/maxrounds', 'presolving/donotaggr', 'presolving/donotmultaggr',
                'separating/oddcycle/freq', 'propagating/maxrounds', 'propagating/maxroundsroot',
                'separating/maxroundsroot', 'separating/maxstallrounds', 'separating/gomory/freq',
                'separating/zerohalf/maxrounds', 'separating/disjunctive/maxroundsroot',
                'cutselection/hybrid/minortho', 'cutselection/ensemble/maxnonzerorootround',
                'heuristics/feaspump/freq', 'heuristics/alns/freq', 'heuristics/undercover/freq',
                'heuristics/dins/minfixingrate', 'heuristics/lpface/minfixingrate',
                'heuristics/subnlp/presolveemphasis', 'branching/preferbinary',
                'branching/fullstrong/maxdepth', 'branching/relpscost/sbiterquot',
                'branching/relpscost/degeneracyaware', 'conflict/enable', 'conflict/useinflp'
            ]

            for rank, (orig_idx, rc) in enumerate(sorted_configs, 1):
                config = rc.get('config', {})
                res = rc.get('result', {})
                exceeded = rc.get('exceeded', False)

                solve_time = res.get('time', float('inf'))
                status = res.get('status', 'unknown')
                speedup = default_time / solve_time if solve_time > 0 else 0

                # Mark best config
                best_marker = " ⭐ **BEST**" if orig_idx == result.get('best_config_index', -1) else ""
                timeout_marker = " ⚠️ **TIMEOUT**" if exceeded else ""

                lines.append(f"#### Config {orig_idx + 1} (Rank {rank}){best_marker}{timeout_marker}\n")
                lines.append(f"**Solve time**: {solve_time:.3f}s (speedup: {speedup:.2f}x), "
                           f"status: {status}\n")

                # Print all 30 parameters
                lines.append("**Parameter values**:\n")
                for param_name in param_names:
                    value = config.get(param_name, 'N/A')
                    lines.append(f"- {param_name}: {value}")
                lines.append("")

        return "\n".join(lines)

    def compress(self):
        """Compression disabled — all experience is kept for LLM reference."""
        pass

    def get_summary(self, max_length: int = 2000) -> str:
        """
        Get a concise summary of the skill file.

        Args:
            max_length: Maximum character length

        Returns:
            Truncated skill content
        """
        content = self.load()
        if len(content) <= max_length:
            return content

        return content[:max_length] + "\n\n[... truncated for brevity ...]"
