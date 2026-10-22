"""
LLM Direction Recommender (Multi-Parameter Version)

Uses LLM API to recommend a 30-dimensional direction vector for a MIP instance.
Output: numpy array of shape (30,)
  - For numeric (int/real) parameters: -1 (decrease), 0 (keep), +1 (increase)
  - For bool/char parameters: 0 (keep default) or 1 (change)
"""

import json
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..features.instance_descriptor import InstanceDescriptor


class DirectionRecommender:
    """
    Use LLM to recommend a 30-dimensional direction vector for MIP instances.
    """

    def __init__(self, llm_client, param_config_path: str, skill_path: str, max_nonzero: int = 30):
        """
        Args:
            llm_client: LLM client instance
            param_config_path: Path to param_config.json
            skill_path: Path to skill.md experience file
            max_nonzero: (unused, kept for backward compatibility)
        """
        self.llm_client = llm_client
        self.param_config_path = param_config_path
        self.skill_path = skill_path

        # Load parameter config
        with open(param_config_path, 'r', encoding='utf-8') as f:
            self.param_config = json.load(f)

        self.n_params = len(self.param_config['parameters'])

        # Instance descriptor for full raw formulation (complete matrix)
        self.descriptor = InstanceDescriptor()

    def batch_recommend(
        self,
        instance_features: List[np.ndarray],
        models: Optional[List] = None,
        max_concurrent: int = 8,
    ) -> List[np.ndarray]:
        """
        Recommend 30-dim direction vectors for multiple instances in parallel.

        Args:
            instance_features: List of 20-dimensional feature vectors
            models: Optional list of PySCIPOpt Model objects for rich description
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of 30-dimensional direction vectors (numpy arrays)
        """
        # Load skill content once
        skill_content = self._load_skill()

        # Build all prompts
        prompts = []
        for idx, features in enumerate(instance_features):
            model = models[idx] if models is not None else None

            # Generate rich description if model is provided
            instance_description = None
            if model is not None:
                instance_description = self.descriptor.describe(model)

            prompt = self._build_prompt(features, skill_content, instance_description)
            prompts.append(prompt)

        # Call API in parallel
        responses = self.llm_client.batch_ask(
            prompts=prompts,
            max_concurrent=max_concurrent,
        )

        # Parse all responses
        results = []
        for response in responses:
            parsed = self._parse_direction(response)
            results.append(parsed)

        return results

    def recommend(self, instance_features: np.ndarray, model=None) -> np.ndarray:
        """
        Recommend a 30-dimensional direction vector for a MIP instance.

        Args:
            instance_features: 20-dimensional feature vector
            model: Optional PySCIPOpt Model for rich description

        Returns:
            30-dimensional numpy array with direction recommendations
        """
        # Load skill file
        skill_content = self._load_skill()

        # Generate rich description if model is provided
        instance_description = None
        if model is not None:
            instance_description = self.descriptor.describe(model)

        # Build prompt
        prompt = self._build_prompt(instance_features, skill_content, instance_description)

        # Call LLM
        response = self.llm_client.ask(prompt)

        # Parse response (returns either 2 or 3 element tuple)
        return self._parse_direction(response)

    def _load_skill(self) -> str:
        """Load skill/experience file content."""
        try:
            with open(self.skill_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "(No experience accumulated yet)"

    def _build_prompt(self, features: np.ndarray, skill: str,
                      instance_description: Optional[str] = None) -> str:
        """Build the prompt for LLM 30-dimensional direction recommendation."""

        # Format parameter list with indices and type-specific encoding rules
        param_list = []
        for i, p in enumerate(self.param_config['parameters']):
            param_type = p['type']
            desc = p.get('description', '')
            # Indicate parameter type for direction encoding
            if param_type in ('int', 'real'):
                type_hint = " [numeric: -1=decrease, 0=keep, +1=increase]"
            else:
                type_hint = " [bool/char: 0=keep default, 1=change]"
            param_list.append(
                f"[{i}] {p['name']}: default={p['default']}, type={param_type}, "
                f"candidates={p['candidates']}, desc=\"{desc}\"{type_hint}"
            )
        param_text = "\n".join(param_list)

        # Format features
        feature_names = [
            "row_count", "col_count", "total_nonzeros", "matrix_density",
            "binary_ratio", "general_integer_ratio", "continuous_ratio", "unbounded_count",
            "equality_ratio", "inequality_ratio", "set_operation_count", "special_constraint_count",
            "mean_nonzeros_per_row", "mean_nonzeros_per_col", "matrix_coeff_cv",
            "objective_density", "rhs_cv",
            "num_connected_components", "largest_component_ratio", "isolated_variable_count"
        ]
        feature_text = "\n".join([f"- {name}: {features[i]:.4f}" for i, name in enumerate(feature_names)])

        # Build instance description section
        if instance_description:
            instance_section = f"""
## Detailed Instance Description
The following provides detailed information about the constraint matrix, variables, and objective function:

{instance_description}
"""
        else:
            instance_section = ""

        prompt = f"""You are an expert in MIP solver parameter tuning.

## Task
Given a MIP instance, analyze its structure and recommend a **30-dimensional direction vector** indicating how each of the 30 SCIP parameters should be adjusted to improve solver performance.

## Step 1: Analyze Problem Structure
You are given TWO views of the instance: (a) 20 aggregate numerical features, and (b) the
**raw formulation** (full variable list, every constraint with its coefficients, and the objective)
in the Detailed Instance Description. The 20 features are **lossy summaries** — they cannot capture
symmetry, special/coupling constraint patterns, coefficient structure, or block structure.

**You MUST read the raw constraint matrix directly, not just the 20 features.** When the Detailed
Instance Description is present, treat it as the PRIMARY source of structural reasoning and use the
20 features only as a quick summary. From the raw formulation, actively look for:
- Variable types and naming patterns — what kind of decisions are being modeled?
- Constraint structure and patterns — covering (≥), packing (≤1), assignment (=1), flow conservation, big-M disjunctions?
- **Symmetry** — interchangeable variables/blocks, repeated coefficient rows, identical columns (critical for tuning symmetry handling)?
- Coefficient distribution and RHS patterns — numerically challenging? big-M present?
- Objective structure — dense or sparse? What is being optimized (min/max)?
- Graph/block topology — network structure, block-diagonal, or fully connected coupling?

Use this raw-matrix analysis to match against patterns in the accumulated experience. Signals you can
only see in the raw matrix (e.g. symmetry, ≤ vs ≥ direction, special structure) should drive parameters
that the aggregate features would otherwise leave undifferentiated.

## Step 2: Recommend Direction for Each Parameter
Based on your problem analysis and accumulated experience, for each of the 30 parameters, recommend:
- For **numeric parameters** (int/real): -1 (decrease), 0 (keep default), or +1 (increase)
- For **bool/char parameters**: 0 (keep default) or 1 (change)

Consider:
- Which parameters have the most impact for this problem structure?
- Should key parameters be increased (more aggressive search) or decreased (reduce overhead)?
- Which parameters should stay at default?
- Parameter interactions — if you increase presolving, should cuts be increased too?

## Output Format
Output a JSON array of exactly 30 elements, one for each parameter (indexed 0-29).

Example:
[1, 0, 1, 0, 0, -1, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]

This means:
- Parameter [0]: increase
- Parameter [1]: keep default
- Parameter [2]: change (bool/char)
- Parameter [5]: decrease
... and so on for all 30 parameters.

## Parameter List (30 parameters, indexed 0-29)
{param_text}

## Numerical Features (20 dimensions)
{feature_text}
{instance_section}
## Accumulated Experience
{skill}

## Instructions
1. Analyze the problem structure from features and detailed description
2. Consult the accumulated experience for similar problem patterns
3. For each of the 30 parameters, recommend a direction
4. Be strategic — focus on parameters that matter most for this problem type
5. Output ONLY the JSON array of 30 integers, no other text

Your response (JSON array of 30 elements only):"""

        return prompt

    def _parse_direction(self, response: str) -> np.ndarray:
        """
        Parse LLM response into a 30-dimensional direction vector.

        Args:
            response: LLM response text containing a JSON array of 30 elements

        Returns:
            numpy array of shape (30,) with validated direction values:
            - For numeric params: -1, 0, or 1
            - For bool/char params: 0 or 1
            Returns all-zeros vector on parse failure.
        """
        # Handle None or empty responses
        if response is None:
            return np.zeros(self.n_params)

        # Try to extract JSON array from response
        response = response.strip()

        # Find array in response
        start = response.find('[')
        end = response.rfind(']') + 1

        if start == -1 or end == 0:
            # No array found, return default
            return np.zeros(self.n_params)

        array_str = response[start:end]

        try:
            result = json.loads(array_str)
        except json.JSONDecodeError:
            # Failed to parse, return default
            return np.zeros(self.n_params)

        # Validate
        if not isinstance(result, list) or len(result) != self.n_params:
            return np.zeros(self.n_params)

        # Parse and validate each element
        direction_vector = np.zeros(self.n_params)
        for i, val in enumerate(result):
            try:
                direction = int(val)
            except (ValueError, TypeError):
                direction = 0

            # Get parameter type for validation
            param_type = self.param_config['parameters'][i]['type']

            if param_type in ('int', 'real'):
                # Numeric: clamp to {-1, 0, 1}
                direction = max(-1, min(direction, 1))
            else:
                # Bool/char: clamp to {0, 1}
                direction = max(0, min(direction, 1))

            direction_vector[i] = direction

        return direction_vector
