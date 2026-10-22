"""
Training Pipeline (Multi-Parameter Version)

Orchestrates the two-phase training process:
Phase 1: LLM Skill Accumulation (random configuration experiments)
Phase 2: XGBoost Training (random search → LLM 30-dim direction → 50-dim XGBoost)
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import yaml
try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        HAS_OPENPYXL = True
except ImportError:
        HAS_OPENPYXL = False

from ..generators import ProblemRegistry
from ..features.instance_features import InstanceFeatureExtractor
from ..features.instance_descriptor import InstanceDescriptor
from ..solver import SCIPWrapper, SolveResult, compute_score, compare_results
from ..llm import LLMClient, DirectionRecommender, SkillManager
from ..model.param_predictor import ParameterPredictor
from ..utils.parallel import ParallelExecutor


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Trainer:
    """
    Single-parameter training pipeline:
    1. Phase 1: Exhaustively test all single-parameter changes, accumulate experience
    2. Phase 2: LLM recommends (param, direction) → search that direction → train XGBoost
    """

    def __init__(self, config_path: str = "config/experiment.yaml"):
        """
        Args:
            config_path: Path to experiment configuration
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.registry = ProblemRegistry()
        self.feature_extractor = InstanceFeatureExtractor()
        self.scip = SCIPWrapper(self._load_param_config())
        self.parallel = ParallelExecutor(max_workers=self.config['solver']['max_workers'])

        # Paths
        self.param_config_path = self.config['paths']['param_config']
        self.skill_path = Path(self.config['paths']['skill_dir']) / "skill.md"
        self.results_dir = Path(self.config['paths']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM components
        llm_config = self.config['llm']
        self.llm_client = LLMClient(
            provider=llm_config['provider'],
            model=llm_config['model'],
            base_url=llm_config.get('base_url'),
            api_key=llm_config.get('api_key_env'),
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens'],
        )

        self.direction_recommender = DirectionRecommender(
            llm_client=self.llm_client,
            param_config_path=self.param_config_path,
            skill_path=str(self.skill_path),
        )

        self.skill_manager = SkillManager(skill_path=str(self.skill_path), llm_client=self.llm_client)

        # Initialize predictor
        self.predictor = ParameterPredictor(param_config_path=self.param_config_path)

    def _load_param_config(self) -> Dict:
        """Load parameter configuration."""
        with open(self.config['paths']['param_config'], 'r', encoding='utf-8') as f:
            return json.load(f)

    def _instance_key(self, family_id: str, variant: str, seed: int) -> str:
        """Generate unique key for an instance."""
        return f"{family_id}/{variant}/{seed}"

    def _get_sort_key(self, value, param_name: str, param_type: str):
        """
        Get sort key for a parameter value, handling -1 semantics correctly.

        For maxrounds/maxdepth parameters: -1 means unlimited (treat as max value)
        For freq parameters: -1 means never (treat as min value)

        Args:
            value: The parameter value
            param_name: Parameter name
            param_type: Parameter type ('int', 'real', etc.)

        Returns:
            Sort key (numeric value for comparison)
        """
        if param_type not in ('int', 'real'):
            return value

        val = float(value) if param_type == 'real' else int(value)

        if val == -1:
            # Check if this is a "unlimited" parameter (maxrounds, maxdepth, maxstallrounds)
            if 'maxrounds' in param_name or 'maxdepth' in param_name or 'maxstallrounds' in param_name:
                return float('inf')  # -1 means unlimited, treat as max
            # Check if this is a "never" parameter (freq)
            elif 'freq' in param_name:
                return float('-inf')  # -1 means never, treat as min

        return val

    def _compute_candidate_directions(self, param_def: Dict) -> List[Tuple[any, int]]:
        """
        Compute all possible (value, direction) pairs for a single parameter.

        For numeric (int/real): take adjacent values above (+1) and below (-1) default
        For bool: flip the value (direction=1)
        For char: all alternative values (direction=1)

        Args:
            param_def: Parameter definition dict with keys: name, type, default, candidates

        Returns:
            List of (value, direction) tuples
        """
        param_type = param_def['type']
        param_name = param_def['name']
        default = param_def['default']
        candidates = param_def['candidates']

        results = []

        if param_type in ('int', 'real'):
            # Sort candidates with proper -1 semantics
            sorted_candidates = sorted(candidates,
                                      key=lambda x: self._get_sort_key(x, param_name, param_type))

            # Find default's position in sorted list
            try:
                if param_type == 'int':
                    default_idx = sorted_candidates.index(int(default))
                else:
                    default_idx = sorted_candidates.index(float(default))
            except ValueError:
                # Default not in candidates, shouldn't happen
                default_idx = -1

            if default_idx >= 0:
                # Value below default (direction=-1)
                if default_idx > 0:
                    results.append((sorted_candidates[default_idx - 1], -1))
                # Value above default (direction=+1)
                if default_idx < len(sorted_candidates) - 1:
                    results.append((sorted_candidates[default_idx + 1], 1))

        elif param_type == 'bool':
            # Flip the boolean value
            other = not default
            results.append((other, 1))

        elif param_type in ('char', 'string'):
            # All alternative values
            for c in candidates:
                if c != default:
                    results.append((c, 1))

        return results

    def _generate_single_param_configs(self) -> List[Dict]:
        """
        Generate all single-parameter change configurations.

        For each of the 30 parameters, generate configs that change ONLY that parameter.

        Returns:
            List of dicts with keys:
                - param_index: int (0-29)
                - param_name: str
                - value: the non-default value
                - direction: int (-1, +1, or 1 for bool/char)
                - config: full config dict (all defaults except this one param)
        """
        param_config = self._load_param_config()
        parameters = param_config['parameters']

        # Build default config
        default_config = {p['name']: p['default'] for p in parameters}

        all_configs = []

        for param_idx, param_def in enumerate(parameters):
            param_name = param_def['name']
            candidate_directions = self._compute_candidate_directions(param_def)

            for value, direction in candidate_directions:
                config = default_config.copy()
                config[param_name] = value

                all_configs.append({
                    'param_index': param_idx,
                    'param_name': param_name,
                    'value': value,
                    'direction': direction,
                    'config': config,
                })

        return all_configs

    def _get_direction_candidates(self, param_index: int, direction: int) -> List[Tuple[any, any]]:
        """
        Get all candidate values in a given direction for a parameter.

        Args:
            param_index: Parameter index (0-29)
            direction: -1 (decrease) or +1 (increase) or 1 (change for bool/char)

        Returns:
            List of (value, config_value) tuples where config_value is what to set in config
        """
        param_config = self._load_param_config()
        param_def = param_config['parameters'][param_index]
        param_name = param_def['name']
        param_type = param_def['type']
        default = param_def['default']
        candidates = param_def['candidates']

        result = []

        if param_type in ('int', 'real'):
            # Use sort key for proper -1 semantics
            default_key = self._get_sort_key(default, param_name, param_type)

            if direction == 1:
                # All candidates greater than default (by sort key)
                for c in candidates:
                    c_key = self._get_sort_key(c, param_name, param_type)
                    if c_key > default_key:
                        result.append((c, c))
            elif direction == -1:
                # All candidates less than default (by sort key)
                for c in candidates:
                    c_key = self._get_sort_key(c, param_name, param_type)
                    if c_key < default_key:
                        result.append((c, c))
        else:
            # bool/char: all candidates different from default
            for c in candidates:
                if c != default:
                    result.append((c, c))

        return result

    def _generate_random_configs(self, n: int = 20, seed: int = 42) -> List[Dict]:
        """
        Generate n random parameter configurations.

        For each of the 30 parameters, independently and uniformly sample
        from all candidates (including default).

        Args:
            n: Number of random configs to generate
            seed: Random seed for reproducibility (should vary per instance)

        Returns:
            List of config dicts, each containing all 30 parameter values
        """
        rng = np.random.RandomState(seed)
        param_config = self._load_param_config()
        parameters = param_config['parameters']

        configs = []
        for _ in range(n):
            config = {}
            for param_def in parameters:
                param_name = param_def['name']
                candidates = param_def['candidates']
                config[param_name] = candidates[rng.randint(len(candidates))]
            configs.append(config)

        return configs

    def _compute_direction_vector(self, config: Dict, default_config: Dict) -> np.ndarray:
        """
        Compute 30-dim direction vector by comparing config to default.

        For numeric params: -1 (decreased), 0 (same), +1 (increased)
        For bool/char params: 0 (same), 1 (changed)

        Args:
            config: The parameter configuration to compare
            default_config: The default parameter configuration

        Returns:
            numpy array of shape (30,) with direction values
        """
        param_config = self._load_param_config()
        parameters = param_config['parameters']

        direction_vector = np.zeros(len(parameters))

        for i, param_def in enumerate(parameters):
            param_name = param_def['name']
            param_type = param_def['type']
            default_val = default_config[param_name]
            config_val = config[param_name]

            if param_type in ('int', 'real'):
                # Numeric: use sort key for proper -1 semantics
                default_key = self._get_sort_key(default_val, param_name, param_type)
                config_key = self._get_sort_key(config_val, param_name, param_type)

                if config_key > default_key:
                    direction_vector[i] = 1
                elif config_key < default_key:
                    direction_vector[i] = -1
                else:
                    direction_vector[i] = 0
            else:
                # Bool/char: 0 if same, 1 if different
                if config_val != default_val:
                    direction_vector[i] = 1
                else:
                    direction_vector[i] = 0

        return direction_vector

        # =========================================================================
        # Phase 1: LLM Skill Accumulation
        # =========================================================================

    def _load_phase1_checkpoint(self, results_path: Path) -> tuple:
        """Load Phase 1 checkpoint."""
        if not results_path.exists():
            return [], set()

        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            completed_keys = {
                self._instance_key(r['family_id'], r['variant'], r['seed'])
                for r in results
            }
            print(f"  Loaded checkpoint: {len(results)} instances already completed")
            return results, completed_keys
        except Exception as e:
            print(f"  Warning: Failed to load checkpoint ({e}), starting fresh")
            return [], set()

    def _save_phase1_checkpoint(self, results_path: Path, results: list):
        """Save Phase 1 results incrementally."""
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

    def _export_runtime_excel(self, results: list, output_path: Path):
        """
        Export Phase 1 runtime data to Excel.
        Rows: default + N random configurations
        Columns: instances
        """
        if not HAS_OPENPYXL:
            print("    Warning: openpyxl not installed, skipping Excel export")
            return

        try:
            wb = Workbook()
            ws = wb.active
            ws.title = "Runtime Table"

            n_instances = len(results)
            # Determine max number of random configs across all results
            max_random = max((len(r.get('random_configs', [])) for r in results), default=0)

            # Header row
            headers = ["Configuration"]
            for result in results:
                inst_id = f"{result['family_id']}/{result['variant']}/{result['seed']}"
                headers.append(inst_id)
            ws.append(headers)

            # Default row
            row_default = ["default"]
            for result in results:
                row_default.append(round(result['default_result']['time'], 4))
            ws.append(row_default)

            # Random config rows
            for cfg_idx in range(max_random):
                label = f"random_{cfg_idx + 1}"
                row = [label]
                for result in results:
                    rcs = result.get('random_configs', [])
                    if cfg_idx < len(rcs):
                        t = rcs[cfg_idx]['result']['time']
                        row.append(round(t, 4))
                    else:
                        row.append(None)
                ws.append(row)

            # Formatting
            header_font = Font(bold=True)
            header_fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
            for cell in ws[1]:
                cell.font = header_font
                cell.fill = header_fill

            red_fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")
            green_fill = PatternFill(start_color="CCFFCC", end_color="CCFFCC", fill_type="solid")

            for row_idx in range(2, max_random + 2):
                for col_idx in range(2, n_instances + 2):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    default_cell = ws.cell(row=2, column=col_idx)

                    if cell.value is not None and default_cell.value is not None:
                        if cell.value > default_cell.value * 1.5:
                            cell.fill = red_fill
                        elif cell.value < default_cell.value * 0.8:
                            cell.fill = green_fill

            # Auto-adjust column widths
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width

            # Save workbook
            output_path.parent.mkdir(parents=True, exist_ok=True)
            wb.save(output_path)
            print(f"    Runtime table exported to {output_path}")

        except Exception as e:
            print(f"    Warning: Failed to export runtime table: {e}")
            import traceback
            traceback.print_exc()

    def _export_instance_details_excel(self, details: list, output_path: Path):
        """
        Export per-instance evaluation details to Excel (one row per instance).
        Columns: idx, family, seed, variant, 20 features, n_changed, rec/def time,
        speedup, score, winner. Green winner=REC, red winner=DEF.
        """
        if not HAS_OPENPYXL:
            print("    Warning: openpyxl not installed, skipping Excel export")
            return
        if not details:
            print("    Warning: no instance details to export")
            return

        try:
            from al_param_tuning.features.instance_features import InstanceFeatureExtractor
            feature_names = InstanceFeatureExtractor.FEATURE_NAMES

            wb = Workbook()
            ws = wb.active
            ws.title = "Instance Details"

            headers = (["idx", "family_id", "seed", "variant"]
                       + list(feature_names)
                       + ["n_params_changed", "rec_time", "def_time",
                          "speedup", "score", "winner"])
            ws.append(headers)

            winner_col = len(headers)  # 1-based index of 'winner'
            for d in details:
                feats = d.get('features', [])
                feats = [round(float(v), 4) for v in feats] if len(feats) else [None] * len(feature_names)
                row = ([d.get('idx'), d.get('family_id'), d.get('seed'), d.get('variant')]
                       + feats
                       + [d.get('n_changed'),
                          round(d.get('rec_time', 0.0), 4),
                          round(d.get('def_time', 0.0), 4),
                          round(d.get('speedup', 0.0), 3),
                          round(d.get('score', 0.0), 4) if d.get('score') is not None else None,
                          d.get('winner')])
                ws.append(row)

            # Header styling
            header_font = Font(bold=True)
            header_fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
            for cell in ws[1]:
                cell.font = header_font
                cell.fill = header_fill

            # Winner cell coloring
            green_fill = PatternFill(start_color="CCFFCC", end_color="CCFFCC", fill_type="solid")
            red_fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")
            for row_idx in range(2, len(details) + 2):
                cell = ws.cell(row=row_idx, column=winner_col)
                if cell.value == "REC":
                    cell.fill = green_fill
                elif cell.value == "DEF":
                    cell.fill = red_fill

            # Auto-size columns
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except Exception:
                        pass
                ws.column_dimensions[column_letter].width = min(max_length + 2, 40)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            wb.save(output_path)
            print(f"    Instance details exported to {output_path}")

        except Exception as e:
            print(f"    Warning: Failed to export instance details: {e}")
            import traceback
            traceback.print_exc()

    def _preload_instances(self, instances: list, phase_name: str) -> Dict[str, str]:
        """
        Pre-generate all instances and save as LP files.

        Each instance is validated by solving with the default config: it is kept
        only if the default solve time falls within [MIN_SOLVE_TIME, MAX_SOLVE_TIME]
        seconds. Otherwise the instance is deterministically resampled (derived seed)
        and regenerated, up to MAX_RESAMPLE_ATTEMPTS times.

        Returns a dict mapping instance_key to LP file path.
        """
        from pyscipopt import Model

        # Solve-time acceptance window and cache dir are config-driven; defaults
        # preserve the original behavior for configs that don't specify them.
        inst_cfg = self.config.get('instances', {}) or {}
        MIN_SOLVE_TIME = float(inst_cfg.get('min_solve_time', 0.1))
        MAX_SOLVE_TIME = float(inst_cfg.get('max_solve_time', 5.0))
        MAX_RESAMPLE_ATTEMPTS = int(inst_cfg.get('max_resample_attempts', 40))

        default_config = self.scip.get_default_config()

        # Create cache directory (config-driven; defaults to shared dir)
        cache_dir = Path(self.config['paths'].get('instance_cache_dir', 'data/results/instances'))
        cache_dir.mkdir(parents=True, exist_ok=True)

        instance_files = {}

        print(f"    Pre-generating {len(instances)} instances (valid window "
              f"[{MIN_SOLVE_TIME}, {MAX_SOLVE_TIME}]s)...")
        for inst in tqdm(instances, desc=f"    Pre-generating ({phase_name})"):
            instance_key = self._instance_key(inst['family_id'], inst['variant'], inst['seed'])
            lp_file = cache_dir / f"{inst['family_id']}_{inst['variant']}_{inst['seed']}.lp"

            # Reuse cached file if present
            if lp_file.exists():
                instance_files[instance_key] = str(lp_file)
                continue

            base_seed = inst['seed']
            accepted_seed = None
            fallback_seed = base_seed          # best-effort seed if none lands in window
            fallback_time = float('inf')       # smallest solve time seen so far
            fallback_has_sol = False
            for attempt in range(MAX_RESAMPLE_ATTEMPTS):
                # attempt 0 uses the original seed; later attempts derive new seeds
                cand_seed = base_seed if attempt == 0 else base_seed * 1000 + attempt
                model = self.registry.generate_instance(
                    family_id=inst['family_id'],
                    variant=inst['variant'],
                    seed=cand_seed,
                )
                result = self.scip.solve(model, default_config, time_limit=MAX_SOLVE_TIME + 2.0)
                # Track the fastest candidate that produced a solution as fallback,
                # so we never keep a timed-out/runaway instance (the bin_packing case).
                cand_better = (
                    (result.has_solution and not fallback_has_sol)
                    or (result.has_solution == fallback_has_sol and result.time < fallback_time)
                )
                if cand_better:
                    fallback_seed = cand_seed
                    fallback_time = result.time
                    fallback_has_sol = result.has_solution
                if result.has_solution and MIN_SOLVE_TIME <= result.time <= MAX_SOLVE_TIME:
                    accepted_seed = cand_seed
                    break

            # Regenerate a fresh (unsolved) model from the chosen seed for writing.
            write_seed = accepted_seed if accepted_seed is not None else fallback_seed
            accepted_model = self.registry.generate_instance(
                family_id=inst['family_id'],
                variant=inst['variant'],
                seed=write_seed,
            )

            accepted_model.writeProblem(str(lp_file))
            instance_files[instance_key] = str(lp_file)

        return instance_files

    def _load_model_from_lp(self, lp_file: str):
        """Load a model from LP file."""
        from pyscipopt import Model
        model = Model()
        model.readProblem(lp_file)
        model.hideOutput()  # 抑制 SCIP 输出
        return model

    def _solve_avg(self, lp_file: str, config, time_limit: float, n_repeats: int = 5):
        """Solve the same instance serially n_repeats times and return a SolveResult
        whose wall-clock time and node count are averaged over the runs (other fields
        from the last run). Used under single-threaded solving to smooth timing jitter."""
        from dataclasses import replace as _dc_replace
        times, node_counts = [], []
        last = None
        for _ in range(max(1, n_repeats)):
            model = self._load_model_from_lp(lp_file)
            r = self.scip.solve(model, config, time_limit=time_limit)
            times.append(r.time)
            node_counts.append(r.nodes)
            last = r
        return _dc_replace(last, time=float(np.mean(times)), nodes=int(np.mean(node_counts)))

    def run_phase1_skill(
        self,
        n_instances: Optional[int] = None,
        n_configs: Optional[int] = None,  # unused, kept for API compat
        fresh: bool = False,
    ) -> Dict:
        """
        Phase 1: LLM Skill Accumulation (Multi-Parameter Random Search)

        For each instance:
        1. Solve default config to get baseline time
        2. Generate n random configs (each param independently sampled from candidates)
        3. Solve all random configs with adaptive timeout
        4. Send ALL configs + results (positive AND negative) to LLM for skill update

        Args:
            n_instances: Override number of instances (from config if None)
            n_configs: (unused, kept for API compatibility)
            fresh: If True, ignore checkpoint and start fresh

        Returns:
            Dict with statistics
        """
        skill_config = self.config['skill']
        n_instances = n_instances or skill_config['n_instances']
        n_random = skill_config.get('n_random_configs', 20)
        seed_offset = skill_config.get('seed_offset', 0)
        families = skill_config['families']
        variants = skill_config['variants']
        time_limit = self.config['solver']['time_limit']

        results_path = self.results_dir / "phase1_results.json"

        print(f"Phase 1: Skill Accumulation (Multi-Parameter Random Search)")
        print(f"  Families: {families}")
        print(f"  Instances per family: {n_instances}")
        print(f"  Seed range: [{seed_offset}, {seed_offset + n_instances})")
        print(f"  Random configs per instance: {n_random}")
        print(f"  Time limit: {time_limit}s")

        # Load checkpoint (unless --fresh)
        all_results = []
        completed_keys = set()
        if not fresh:
            all_results, completed_keys = self._load_phase1_checkpoint(results_path)
        else:
            print("  Starting fresh (--fresh flag)")

        # Generate instance list
        instances = []
        for family_id in families:
            for variant in variants:
                for seed in range(seed_offset, seed_offset + n_instances):
                    instances.append({
                        'family_id': family_id,
                        'variant': variant,
                        'seed': seed,
                    })

        # Filter out already-completed instances
        pending_instances = [
            inst for inst in instances
            if self._instance_key(inst['family_id'], inst['variant'], inst['seed']) not in completed_keys
        ]

        print(f"  Total instances: {len(instances)}")
        print(f"  Already completed: {len(completed_keys)}")
        print(f"  Remaining to process: {len(pending_instances)}")

        if not pending_instances:
            print("  All instances already processed. Skipping solving.")
            # Rebuild batch_skill_updates from checkpoint
            instance_files = self._preload_instances(instances, "Phase 1 (reload)")
            batch_skill_updates = []
            for result in all_results:
                instance_key = self._instance_key(result['family_id'], result['variant'], result['seed'])
                lp_file = instance_files.get(instance_key)
                if lp_file:
                    model = self._load_model_from_lp(lp_file)
                    entry = {**result, 'model': model}
                    batch_skill_updates.append(entry)
        else:
            # Step 1: Pre-generate instances and extract features
            print("    Pre-generating instances...")
            instance_files = self._preload_instances(pending_instances, "Phase 1")

            print("    Extracting features...")
            phase1_features_list = []
            phase1_models_list = []
            phase1_instances_info = []
            for instance_info in tqdm(pending_instances, desc="    Feature extraction"):
                instance_key = self._instance_key(instance_info['family_id'], instance_info['variant'], instance_info['seed'])
                lp_file = instance_files[instance_key]

                model = self._load_model_from_lp(lp_file)
                features = self.feature_extractor.extract(model)

                phase1_features_list.append(features)
                phase1_models_list.append(model)
                phase1_instances_info.append(instance_info)

            # Step 2: Solve random configs per instance
            from concurrent.futures import ThreadPoolExecutor, as_completed
            print("    Solving random configurations (parallel)...")

            default_config = self.scip.get_default_config()
            batch_skill_updates = []

            # Resume checkpoint-loaded instances for skill updates
            if all_results:
                completed_infos = [
                    {'family_id': r['family_id'], 'variant': r['variant'], 'seed': r['seed']}
                    for r in all_results
                ]
                checkpoint_files = self._preload_instances(completed_infos, "Phase 1 (reload checkpoint)")
                for result in all_results:
                    key = self._instance_key(result['family_id'], result['variant'], result['seed'])
                    lp_file = checkpoint_files.get(key)
                    if lp_file:
                        model = self._load_model_from_lp(lp_file)
                        batch_skill_updates.append({**result, 'model': model})

            for idx in tqdm(range(len(pending_instances)), desc="    Solving"):
                instance_info = phase1_instances_info[idx]
                instance_key = self._instance_key(instance_info['family_id'], instance_info['variant'], instance_info['seed'])
                lp_file = instance_files[instance_key]
                features = phase1_features_list[idx]

                # Solve default config: single-threaded, single run
                default_result = self._solve_avg(lp_file, default_config, time_limit, n_repeats=1)

                # Adaptive time limit: 1.5× default time, floored at 0.5s, capped at 10s
                adaptive_time_limit = min(max(default_result.time * 1.5, 0.5), 10.0)

                # Generate random configs with deterministic seed per instance
                instance_seed = hash(instance_key) % (2**31)
                random_configs = self._generate_random_configs(n=n_random, seed=instance_seed)

                # Solve all random configs SERIALLY (single-thread), single run each
                random_results = []
                for rc in random_configs:
                    result = self._solve_avg(lp_file, rc, adaptive_time_limit, n_repeats=1)
                    exceeded = (result.time > adaptive_time_limit * 1.1)
                    random_results.append({'config': rc, 'result': result, 'exceeded': exceeded})

                # Find fastest config
                best_config_index = -1
                best_time = float('inf')
                best_config = default_config.copy()

                for i, rr in enumerate(random_results):
                    if rr['result'].time < best_time:
                        best_time = rr['result'].time
                        best_config_index = i
                        best_config = rr['config']

                best_speedup = default_result.time / best_time if best_time > 0 else 0

                # Build serializable random_configs list
                random_configs_serializable = []
                for rr in random_results:
                    res = rr['result']
                    random_configs_serializable.append({
                        'config': rr['config'],
                        'result': {
                            'status': res.status,
                            'objective': res.objective if res.has_solution else None,
                            'time': res.time,
                            'gap': res.gap if res.has_solution else None,
                            'has_solution': res.has_solution,
                            'is_optimal': res.is_optimal,
                        },
                        'exceeded': rr.get('exceeded', False),
                    })

                result_entry = {
                    'family_id': instance_info['family_id'],
                    'variant': instance_info['variant'],
                    'seed': instance_info['seed'],
                    'features': features.tolist(),
                    'random_configs': random_configs_serializable,
                    'default_result': {
                        'status': default_result.status,
                        'objective': default_result.objective if default_result.has_solution else None,
                        'time': default_result.time,
                        'gap': default_result.gap if default_result.has_solution else None,
                        'has_solution': default_result.has_solution,
                        'is_optimal': default_result.is_optimal,
                    },
                    'best_config': best_config,
                    'best_config_index': best_config_index,
                    'best_time': best_time,
                    'best_speedup': best_speedup,
                }
                all_results.append(result_entry)

                # Collect for batch skill update (include model for rich description)
                batch_skill_updates.append({**result_entry, 'model': phase1_models_list[idx]})

                # Incremental save
                self._save_phase1_checkpoint(results_path, all_results)
                print(f"    [Progress: {len(all_results)}/{len(instances)} total, "
                      f"best_speedup={best_speedup:.2f}x, adaptive_timeout={adaptive_time_limit:.1f}s]")

        # Step 3: Batch update skill file
        if batch_skill_updates:
            print(f"    Updating skill file with {len(batch_skill_updates)} instances "
                  f"(parallel, max_concurrent=8)...")
            self.skill_manager.batch_update(batch_skill_updates, max_concurrent=8)

        # Export results to Excel
        print(f"    Exporting results to Excel...")
        self._export_runtime_excel(all_results, self.results_dir / "phase1_runtime.xlsx")

        # Final save
        self._save_phase1_checkpoint(results_path, all_results)
        print(f"  Phase 1 complete. Results saved to {results_path}")

        speedups = [r['best_speedup'] for r in all_results]
        return {
            'n_instances': len(all_results),
            'mean_best_speedup': np.mean(speedups) if speedups else 0.0,
            'mean_time': np.mean([
                r['default_result']['time'] for r in all_results
            ]) if all_results else 0.0,
        }

        # =========================================================================
        # Phase 2: XGBoost Training
        # =========================================================================

    def run_phase2_xgboost(
        self,
        n_instances: Optional[int] = None,
        n_configs: Optional[int] = None,  # unused, kept for API compat
        force_refresh: bool = False,
        use_phase1: bool = True,  # unused, kept for API compat
        phase1_only: bool = False,  # unused, kept for API compat
        reuse_directions: bool = False,
    ) -> Dict:
        """
        Phase 2: XGBoost Training (Multi-Parameter)

        For each instance:
        1. Extract 20-dim features
        2. Solve default + 20 random configs → fastest config → labels for all 30 params
        3. LLM recommends 30-dim direction vector (or reuse from old cache)
        4. Construct 50-dim input = [features, direction_vector]
        5. Train 30 XGBoost classifiers (one per parameter)

        Args:
            n_instances: Override number of instances (from config if None)
            force_refresh: If True, ignore cache and regenerate training data
            reuse_directions: If True, reuse LLM direction vectors from old cache,
                             but re-solve random configs

        Returns:
            Dict with training statistics
        """
        train_config = self.config['training']
        n_instances = n_instances or train_config['n_instances']
        n_random = train_config.get('n_random_configs', 20)
        seed_offset = train_config.get('seed_offset', 0)
        families = train_config['families']
        variants = train_config['variants']
        time_limit = train_config['time_limit']

        print(f"\nPhase 2: XGBoost Training (Multi-Parameter)")
        print(f"  Families: {families}")
        print(f"  Instances per family: {n_instances}")
        print(f"  Seed range: [{seed_offset}, {seed_offset + n_instances})")
        print(f"  Random configs per instance: {n_random}")
        print(f"  Time limit: {time_limit}s")

        # Generate instance list
        instances = []
        for family_id in families:
            for variant in variants:
                for seed in range(seed_offset, seed_offset + n_instances):
                    instances.append({
                        'family_id': family_id,
                        'variant': variant,
                        'seed': seed,
                    })

        # Check for cached training data
        cache_path = Path(train_config.get('cache_path', self.results_dir / "training_cache_multi.json"))
        cache_sig = self._get_training_cache_signature(train_config)

        cache_data = None
        if not force_refresh:
            cache_data = self._load_training_cache(cache_path, cache_sig)

        # If reuse_directions=True and force_refresh=True, load old cache just for directions
        old_cache_for_directions = None
        if reuse_directions and force_refresh and cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    old_cache_for_directions = json.load(f)
                print(f"  Loaded old cache for direction reuse ({len(old_cache_for_directions.get('X_all', []))} instances)")
            except Exception as e:
                print(f"  Warning: Failed to load old cache for directions ({e})")

        # Training data structures
        # X_all: list of 50-dim input vectors (shared across all params)
        # y_all: dict of param_name -> list of label values (fastest config's value for that param)
        X_all = []  # list of 50-dim numpy arrays
        y_all = {}  # param_name -> list of values
        raw_data_all = []  # per-instance: {'default_time': float, 'configs': [{'time': float, 'values': {param: val}}]}
        phase2_completed_keys = set()

        # Cache features & directions for reuse in training set evaluation
        # Dict keyed by instance_key -> (features, direction_vector)
        cached_feat_dir = {}

        if cache_data is not None and cache_data.get('complete', False):
            print("  Using cached training data (complete)...")
            X_all = [np.array(x) for x in cache_data['X_all']]
            y_all = {k: list(v) for k, v in cache_data['y_all'].items()}
            raw_data_all = cache_data.get('raw_data_all', [])
            print(f"  Loaded: {len(X_all)} instances, {len(y_all)} params")
            if raw_data_all:
                print(f"  Loaded raw config data for {len(raw_data_all)} instances (weighted training enabled)")
            else:
                print(f"  WARNING: no raw_data_all in cache — weighted training unavailable, falling back to best-of-20")

            # Populate cached_feat_dir so training-set evaluation can reuse directions without LLM calls
            for inst, x in zip(instances, X_all):
                inst_key = self._instance_key(inst['family_id'], inst['variant'], inst['seed'])
                cached_feat_dir[inst_key] = (x[:20].copy(), x[20:].copy())
            print(f"  Cached features+directions populated for {len(cached_feat_dir)} instances (training-set eval will reuse)")
        else:
            print("  Generating training data...")

            # Resume from partial cache
            if cache_data is not None and not cache_data.get('complete', False):
                X_all = [np.array(x) for x in cache_data.get('X_all', [])]
                y_all = {k: list(v) for k, v in cache_data.get('y_all', {}).items()}
                raw_data_all = cache_data.get('raw_data_all', [])
                phase2_completed_keys = set(cache_data.get('phase2_completed_keys', []))
                print(f"  Resuming from partial cache: {len(X_all)} instances completed")

            # Filter out already-completed instances
            pending_instances = [
                inst for inst in instances
                if self._instance_key(inst['family_id'], inst['variant'], inst['seed']) not in phase2_completed_keys
            ]
            print(f"  Total instances: {len(instances)}, Completed: {len(phase2_completed_keys)}, Pending: {len(pending_instances)}")

            # Step 1: Pre-generate instances and extract features
            print("    Pre-generating instances...")
            instance_files = self._preload_instances(pending_instances, "Phase 2")

            print("    Extracting features...")
            pending_features = []
            pending_models = []
            for instance_info in tqdm(pending_instances, desc="    Feature extraction"):
                instance_key = self._instance_key(instance_info['family_id'], instance_info['variant'], instance_info['seed'])
                lp_file = instance_files[instance_key]
                model = self._load_model_from_lp(lp_file)
                features = self.feature_extractor.extract(model)
                pending_features.append(features)
                pending_models.append(model)

            # Step 2: Batch get LLM 30-dim direction recommendations
            # If reuse_directions=True, extract from old cache instead of calling LLM
            directions_source = cache_data if cache_data else old_cache_for_directions
            if reuse_directions and directions_source is not None and directions_source.get('X_all'):
                old_X_all = directions_source['X_all']

                # Validate: check instance count matches
                if len(old_X_all) != len(pending_instances):
                    print(f"    ERROR: Old cache has {len(old_X_all)} instances, but need {len(pending_instances)}")
                    print(f"    Cannot reuse directions - instance count mismatch")
                    print(f"    Falling back to LLM recommendations...")
                    reuse_directions = False
                else:
                    print("    Reusing LLM direction vectors from old cache...")
                    print(f"    First 3 pending instances:")
                    for i in range(min(3, len(pending_instances))):
                        inst = pending_instances[i]
                        print(f"      [{i}] {inst['family_id']}/{inst['variant']}/{inst['seed']}")

                    # Extract last 30 dims (direction vector) from each X_all entry
                    pending_directions = []
                    for i in range(len(pending_instances)):
                        old_x = old_X_all[i]
                        direction = np.array(old_x[20:])  # last 30 dims
                        pending_directions.append(direction)
                    print(f"    Reused {len(pending_directions)} direction vectors from cache")

            if not reuse_directions or not (directions_source is not None and directions_source.get('X_all')):
                print("    Getting LLM 30-dim direction recommendations (8 parallel)...")
                pending_directions = self.direction_recommender.batch_recommend(
                    instance_features=pending_features,
                    models=pending_models,
                    max_concurrent=8,
                )
                print(f"    Got {len(pending_directions)} direction vectors")

            # Step 3: For each instance, solve random configs and collect training data
            from concurrent.futures import ThreadPoolExecutor, as_completed
            default_config = self.scip.get_default_config()
            param_config = self._load_param_config()
            param_names = [p['name'] for p in param_config['parameters']]

            for idx in tqdm(range(len(pending_instances)), desc="    Solving & collecting"):
                instance_info = pending_instances[idx]
                instance_key = self._instance_key(instance_info['family_id'], instance_info['variant'], instance_info['seed'])
                lp_file = instance_files[instance_key]
                features = pending_features[idx]
                direction_vector = pending_directions[idx]

                # Solve default config: single-threaded, single run
                model_def = self._load_model_from_lp(lp_file)
                default_result = self.scip.solve(model_def, default_config, time_limit=time_limit)

                # Adaptive time limit: 1.5× default time, floored at 0.5s, capped at 10s
                adaptive_time_limit = min(max(default_result.time * 1.5, 0.5), 10.0)

                # Generate and solve random configs with deterministic seed
                instance_seed = hash(instance_key) % (2**31)
                random_configs = self._generate_random_configs(n=n_random, seed=instance_seed)

                # Solve all random configs SERIALLY (single-thread), single run each
                random_results = []
                for rc in random_configs:
                    m = self._load_model_from_lp(lp_file)
                    result = self.scip.solve(m, rc, time_limit=adaptive_time_limit)
                    random_results.append({'config': rc, 'result': result})

                # Find fastest config
                best_time = float('inf')
                best_config = default_config.copy()

                for rr in random_results:
                    if rr['result'].time < best_time:
                        best_time = rr['result'].time
                        best_config = rr['config']

                # Record raw data for weighted training (all configs + times + default time)
                raw_data_all.append({
                    'default_time': default_result.time,
                    'configs': [
                        {
                            'time': rr['result'].time,
                            'values': {pn: rr['config'].get(pn, default_config[pn]) for pn in param_names},
                        }
                        for rr in random_results
                    ],
                })

                # Construct 50-dim input: [features(20), direction_vector(30)]
                X_input = np.concatenate([features, direction_vector])
                X_all.append(X_input)

                # Cache features and directions for training set evaluation reuse
                cached_feat_dir[instance_key] = (features, direction_vector)

                # Extract labels: fastest config's value for each parameter
                for p_name in param_names:
                    if p_name not in y_all:
                        y_all[p_name] = []
                    y_all[p_name].append(best_config.get(p_name, default_config[p_name]))

                best_speedup = default_result.time / best_time if best_time > 0 else 0
                n_changed = sum(1 for p_name in param_names
                               if best_config.get(p_name) != default_config.get(p_name))
                print(f"    [{idx+1}/{len(pending_instances)}] {instance_info['family_id']}/{instance_info['seed']}: "
                      f"best_speedup={best_speedup:.2f}x, "
                      f"fastest_config_changed={n_changed}/30, "
                      f"llm_direction_nonzero={int(np.sum(direction_vector != 0))}/30")

                # Mark as completed and save incrementally
                phase2_completed_keys.add(self._instance_key(instance_info['family_id'], instance_info['variant'], instance_info['seed']))
                self._save_training_cache(cache_path, cache_sig, X_all, y_all, phase2_completed_keys, complete=False, raw_data_all=raw_data_all)

            # Final save
            self._save_training_cache(cache_path, cache_sig, X_all, y_all, phase2_completed_keys, complete=True, raw_data_all=raw_data_all)
            print(f"  Training cache saved to {cache_path}")

        # Train predictors (three-tier strategy) for all 30 params
        print(f"\n  Training predictors (three-tier strategy)...")
        print(f"  Total instances: {len(X_all)}")

        X_array = np.array(X_all)  # shape: (n_instances, 50)

        accuracies = {}
        constant_count = 0
        bool_flip_count = 0

        param_config_full = self._load_param_config()

        # Decide training mode: weighted (all configs faster than default) vs best-of-20 fallback
        use_weighted = bool(raw_data_all) and len(raw_data_all) == len(X_all)
        if use_weighted:
            print(f"  Training mode: WEIGHTED (all configs faster than default, weight = 1 + log(speedup))")
        else:
            print(f"  Training mode: best-of-20 (unweighted) — raw_data_all unavailable")

        def build_weighted(param_name):
            """Build (X, y, w) using every config strictly faster than default.
            Returns None if no qualifying samples."""
            Xw, yw, ww = [], [], []
            for i, rec in enumerate(raw_data_all):
                dt = rec.get('default_time', float('inf'))
                if dt <= 0:
                    continue
                xi = X_all[i]
                for cfg in rec.get('configs', []):
                    t = cfg.get('time', float('inf'))
                    if t <= 0 or t >= dt:
                        continue  # only keep configs strictly faster than default
                    Xw.append(xi)
                    yw.append(cfg['values'].get(param_name))
                    ww.append(1.0 + np.log(dt / t))  # 1 + log(speedup)
            if not Xw:
                return None
            return np.array(Xw), np.array(yw), np.array(ww)

        for param_def in param_config_full['parameters']:
            param_name = param_def['name']
            param_type = param_def['type']

            if param_name not in y_all:
                continue

            # Build training data for this parameter
            if use_weighted:
                built = build_weighted(param_name)
                if built is None:
                    # No config beat default across all instances → fall back to best-of-20 label
                    X_train = X_array
                    y = np.array(y_all[param_name])
                    sample_weight = None
                else:
                    X_train, y, sample_weight = built
            else:
                X_train = X_array
                y = np.array(y_all[param_name])
                sample_weight = None

            unique_labels = np.unique(y)
            n_samples = len(y)

            # Tier 1: XGBoost (2+ unique labels)
            if len(unique_labels) >= 2:
                try:
                    acc = self.predictor.train(X_train, y, param_name, sample_weight=sample_weight)
                    accuracies[param_name] = acc
                    wtag = f", weighted" if sample_weight is not None else ""
                    print(f"    [XGBoost]  {param_name:50s}: {acc:.3f} accuracy ({n_samples} samples, {len(unique_labels)} unique{wtag})")
                except Exception as e:
                    print(f"    [XGBoost]  {param_name:50s}: training failed - {e}")

            # Tier 2: Constant (all labels are the same)
            elif len(unique_labels) == 1:
                constant_value = unique_labels[0]
                self.predictor.set_constant(param_name, constant_value)
                constant_count += 1
                print(f"    [Constant] {param_name:50s}: always predict {constant_value} ({n_samples} samples)")

        # Register all bool params without training data as bool_flip
        for p in param_config_full['parameters']:
            if p['type'] == 'bool' and p['name'] not in accuracies and p['name'] not in self.predictor.constant_params:
                self.predictor.set_bool_flip(p['name'])
                bool_flip_count += 1

        # Save model
        model_save_path = self.config['training']['model_save_path']
        self.predictor.save(model_save_path)
        print(f"\n  Model saved to {model_save_path}")

        # Diagnostics
        print("\n  " + "=" * 60)
        print("  Training Diagnostics (Three-Tier Strategy)")
        print("  " + "=" * 60)

        total_predictors = len(accuracies) + constant_count + bool_flip_count
        print(f"\n  Total predictors: {total_predictors}/30")
        print(f"    Tier 1 (XGBoost):   {len(accuracies)}")
        print(f"    Tier 2 (Constant):  {constant_count}")
        print(f"    Tier 3 (Bool flip): {bool_flip_count}")

        if accuracies:
            sorted_params = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
            print(f"\n  XGBoost Parameters ({len(accuracies)}):")
            for param, acc in sorted_params:
                n_samples = len(y_all.get(param, []))
                print(f"    {param:50s}: {acc:.3f} ({n_samples} samples)")

            acc_values = list(accuracies.values())
            print(f"\n  XGBoost Accuracy Distribution:")
            print(f"    Mean:   {np.mean(acc_values):.3f}")
            print(f"    Median: {np.median(acc_values):.3f}")
            print(f"    Min:    {np.min(acc_values):.3f}")
            print(f"    Max:    {np.max(acc_values):.3f}")

        if constant_count > 0:
            print(f"\n  Constant Predictors ({constant_count}):")
            for param_name, value in self.predictor.constant_params.items():
                print(f"    {param_name:50s}: always {value}")

        if bool_flip_count > 0:
            print(f"\n  Bool Flip Predictors ({bool_flip_count}):")
            for param_name in sorted(self.predictor.bool_flip_params):
                print(f"    {param_name:50s}: flip to other value")


        # =====================================================================
        # Training Set Evaluation
        # =====================================================================
        print("\n  " + "=" * 60)
        print("  Training Set Evaluation")
        print("  " + "=" * 60)

        eval_time_limit = self.config['training']['time_limit']
        default_config = self.scip.get_default_config()
        param_names = [p['name'] for p in param_config_full['parameters']]

        # Re-load all training instances
        print("    Loading training instances...")
        eval_instance_files = self._preload_instances(instances, "Phase 2 Eval")

        # Extract features and get LLM directions — reuse cache when available
        print("    Extracting features...")
        eval_features_list = []
        need_llm_indices = []  # indices of instances that need LLM recommendation
        need_llm_models = {}   # idx -> PySCIPOpt Model (full formulation for LLM)

        for idx, instance_info in enumerate(tqdm(instances, desc="    Features")):
            inst_key = self._instance_key(instance_info['family_id'], instance_info['variant'], instance_info['seed'])

            if inst_key in cached_feat_dir:
                # Reuse cached features and direction from training phase
                features, _ = cached_feat_dir[inst_key]
                eval_features_list.append(features)
            else:
                # Need to extract features (and later get LLM direction)
                lp_file = eval_instance_files[inst_key]
                model = self._load_model_from_lp(lp_file)
                features = self.feature_extractor.extract(model)
                eval_features_list.append(features)
                need_llm_indices.append(idx)
                need_llm_models[idx] = model

        # Get LLM directions only for instances not already cached
        eval_directions = [None] * len(instances)

        # Fill in cached directions first
        for idx, instance_info in enumerate(instances):
            inst_key = self._instance_key(instance_info['family_id'], instance_info['variant'], instance_info['seed'])
            if inst_key in cached_feat_dir:
                _, direction = cached_feat_dir[inst_key]
                eval_directions[idx] = direction

        # Get new LLM recommendations for uncached instances
        if need_llm_indices:
            uncached_features = [eval_features_list[i] for i in need_llm_indices]
            uncached_models = [need_llm_models[i] for i in need_llm_indices]
            print(f"    Getting LLM directions for {len(uncached_features)} uncached instances...")
            new_directions = self.direction_recommender.batch_recommend(
                instance_features=uncached_features,
                models=uncached_models,
                max_concurrent=8,
            )
            for i, orig_idx in enumerate(need_llm_indices):
                eval_directions[orig_idx] = new_directions[i]
        else:
            print("    All LLM directions reused from cache (no new LLM calls)")

        # Evaluate each instance
        from concurrent.futures import ThreadPoolExecutor
        train_scores = []
        train_rec_times = []
        train_def_times = []
        train_details = []

        print("    Evaluating on training set...")
        for idx in tqdm(range(len(instances)), desc="    Training eval"):
            instance_info = instances[idx]
            inst_key = self._instance_key(instance_info['family_id'], instance_info['variant'], instance_info['seed'])
            lp_file = eval_instance_files[inst_key]
            features = eval_features_list[idx]
            direction_vector = eval_directions[idx]

            # Predict full config
            predicted_config = self.predictor.predict(features, direction_vector)

            # Solve predicted vs default serially (single-thread, single run)
            model_rec = self._load_model_from_lp(lp_file)
            result_rec = self.scip.solve(model_rec, predicted_config, time_limit=eval_time_limit)
            model_def = self._load_model_from_lp(lp_file)
            result_def = self.scip.solve(model_def, default_config, time_limit=eval_time_limit)

            score = compute_score(result_rec, result_def)
            train_scores.append(score)
            train_rec_times.append(result_rec.time)
            train_def_times.append(result_def.time)

            n_changed = sum(1 for p_name in param_names
                           if predicted_config.get(p_name) != default_config.get(p_name))
            speedup = result_def.time / max(result_rec.time, 0.001) if result_rec.time > 0 else 0
            winner = "REC" if score and score > 0 else ("DEF" if score and score < 0 else "TIE")
            train_details.append({
                'idx': idx,
                'family_id': instance_info['family_id'],
                'seed': instance_info['seed'],
                'variant': instance_info['variant'],
                'features': list(features) if features is not None else [],
                'n_changed': n_changed,
                'rec_time': result_rec.time,
                'def_time': result_def.time,
                'speedup': speedup,
                'score': score,
                'winner': winner,
            })
            print(f"      {instance_info['family_id']}/{instance_info['seed']}: "
                  f"{n_changed} params changed, speedup={speedup:.2f}x, winner={winner}")

        # Training set summary
        valid_scores = [s for s in train_scores if s is not None]
        n_rec_win = sum(1 for s in valid_scores if s > 0)
        n_def_win = sum(1 for s in valid_scores if s < 0)
        n_tie = sum(1 for s in valid_scores if s == 0)
        win_rate = 100.0 * n_rec_win / len(valid_scores) if valid_scores else 0

        avg_rec = np.mean(train_rec_times)
        avg_def = np.mean(train_def_times)
        overall_speedup = avg_def / max(avg_rec, 0.001)

        print(f"\n  Training Set Results:")
        print(f"    Win rate: {win_rate:.1f}% ({n_rec_win}W / {n_def_win}L / {n_tie}T)")
        print(f"    Mean score: {np.mean(valid_scores):.3f}")
        print(f"    Speedup: {overall_speedup:.2f}x (mean rec={avg_rec:.3f}s, mean def={avg_def:.3f}s)")

        # Per-family breakdown
        print(f"\n    Per-Family:")
        print(f"    {'Family':<25s} {'Win Rate':>10s} {'Speedup':>10s} {'Count':>6s}")
        print(f"    {'-'*25} {'-'*10} {'-'*10} {'-'*6}")
        for family_id in families:
            fam_scores = []
            fam_rec = []
            fam_def = []
            for i, inst in enumerate(instances):
                if inst['family_id'] == family_id:
                    if train_scores[i] is not None:
                        fam_scores.append(train_scores[i])
                    fam_rec.append(train_rec_times[i])
                    fam_def.append(train_def_times[i])
            if fam_scores:
                fam_wr = 100.0 * sum(1 for s in fam_scores if s > 0) / len(fam_scores)
                fam_sp = np.mean(fam_def) / max(np.mean(fam_rec), 0.001)
                print(f"    {family_id:<25s} {fam_wr:>9.1f}% {fam_sp:>9.2f}x {len(fam_scores):>6d}")

        # Export per-instance details to Excel
        self._export_instance_details_excel(train_details, self.results_dir / "phase2_training_details.xlsx")

        return {
            'n_instances': len(instances),
            'n_trained': len(X_all),
            'n_xgboost_params': len(accuracies),
            'n_constant_params': constant_count,
            'n_bool_flip_params': bool_flip_count,
            'n_trained_params': total_predictors,
            'mean_accuracy': np.mean(list(accuracies.values())) if accuracies else 0.0,
            'accuracies': accuracies,
            'train_win_rate': win_rate,
            'train_speedup': overall_speedup,
        }

    def _get_training_cache_signature(self, train_config: Dict) -> str:
        """Generate a signature for training configuration to validate cache."""
        import hashlib
        sig_data = {
            'seed_offset': train_config.get('seed_offset', 0),
            'n_instances': train_config['n_instances'],
            'families': train_config['families'],
            'variants': train_config['variants'],
            'time_limit': train_config['time_limit'],
            'version': 'multi_param_v1',
        }
        sig_str = json.dumps(sig_data, sort_keys=True)
        return hashlib.md5(sig_str.encode()).hexdigest()

    def _load_training_cache(self, cache_path: Path, expected_sig: str) -> Optional[Dict]:
        """Load training data from cache if valid."""
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            if cache_data.get('signature') != expected_sig:
                print(f"  Cache signature mismatch, ignoring cache")
                return None

            print(f"  Loaded training cache: {len(cache_data.get('X_all', []))} instances")
            return cache_data

        except Exception as e:
            print(f"  Warning: Failed to load training cache ({e})")
            return None

    def _save_training_cache(self, cache_path: Path, cache_sig: str, X_all: list,
                             y_all: Dict, phase2_completed_keys: set,
                             complete: bool = False, raw_data_all: list = None):
        """Save training cache incrementally."""
        cache_data_to_save = {
            'signature': cache_sig,
            'complete': complete,
            'X_all': [x.tolist() if hasattr(x, 'tolist') else x for x in X_all],
            'y_all': {k: [str(v) for v in vals] for k, vals in y_all.items()},
            'raw_data_all': raw_data_all if raw_data_all is not None else [],
            'phase2_completed_keys': list(phase2_completed_keys),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Retry logic for file locking
        max_retries = 3
        retry_delay = 1.0
        for attempt in range(max_retries):
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data_to_save, f, indent=2, cls=NumpyEncoder)
                return  # Success
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"    Warning: Cache file locked, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"    Warning: Failed to save cache after {max_retries} attempts: {e}")
                    print(f"    Training will continue, but cache won't be updated.")
                    print(f"    Close any programs that might have '{cache_path}' open.")
                    return

    def run_full_pipeline(self) -> Dict:
        """Run complete training pipeline (Phase 1 + Phase 2)."""
        print("=" * 60)
        print("Starting Full Training Pipeline (Multi-Parameter)")
        print("=" * 60)

        phase1_stats = self.run_phase1_skill()
        phase2_stats = self.run_phase2_xgboost()

        print("\n" + "=" * 60)
        print("Training Pipeline Complete")
        print("=" * 60)
        print(f"Phase 1 - Instances: {phase1_stats['n_instances']}, "
              f"Mean Best Speedup: {phase1_stats['mean_best_speedup']:.2f}x")
        print(f"Phase 2 - Instances: {phase2_stats['n_instances']}, "
              f"Trained Params: {phase2_stats['n_trained_params']}, "
              f"Mean Accuracy: {phase2_stats['mean_accuracy']:.3f}")

        return {
            'phase1': phase1_stats,
            'phase2': phase2_stats,
        }
