"""
Parameter Predictor (Multi-Parameter Version)

Three-tier prediction strategy:
1. XGBoost classifier: when 2+ unique labels in training data
2. Constant predictor: when all training labels are the same
3. Bool flip: for boolean parameters, just flip to the other value
4. Default fallback: no training data at all

Input: 20-dimensional features + 30-dimensional direction vector = 50 dimensions
Output: Predicted values for ALL 30 parameters
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xgboost import XGBClassifier


class ParameterPredictor:
    """
    Predict optimal parameter values for all 30 parameters using a three-tier strategy.
    """

    def __init__(self, param_config_path: str):
        """
        Args:
            param_config_path: Path to param_config.json
        """
        with open(param_config_path, 'r', encoding='utf-8') as f:
            self.param_config = json.load(f)

        self.n_params = len(self.param_config['parameters'])
        self.feature_dim = 20
        self.input_dim = self.feature_dim + self.n_params  # 50 (20 features + 30 directions)

        # Tier 1: XGBoost classifiers (for params with 2+ unique labels)
        self.classifiers: Dict[str, XGBClassifier] = {}

        # Label mappings for XGBoost classifiers (index -> original label value)
        self.label_mappings: Dict[str, Dict[int, any]] = {}

        # Tier 2: Constant predictors (for params where all labels are the same)
        self.constant_params: Dict[str, any] = {}

        # Tier 3: Bool flip params (boolean params → always flip)
        self.bool_flip_params: set = set()

        # Default XGBoost hyperparameters
        self.default_xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'random_state': 42,
        }

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_name: str,
        xgb_params: Optional[Dict] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Train classifier for a single parameter (only when 2+ unique labels).

        Args:
            X: Training features, shape (n_samples, 50) — 20 features + 30 directions
            y: Training labels, shape (n_samples,), values are parameter values
            param_name: Name of the parameter being trained
            xgb_params: Optional XGBoost hyperparameters override
            sample_weight: Optional sample weights, shape (n_samples,)

        Returns:
            Training accuracy
        """
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {X.shape[1]}")

        # Get parameter definition
        param_def = self._get_param_def(param_name)
        candidates = param_def['candidates']

        # Get unique labels in training data
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(f"Need at least 2 unique labels, got {len(unique_labels)}")

        # Map labels to indices (only for labels present in training data)
        label_to_idx = {str(v): i for i, v in enumerate(unique_labels)}
        y_encoded = np.array([label_to_idx[str(v)] for v in y])

        # Create and train classifier
        params = self.default_xgb_params.copy()
        if xgb_params:
            params.update(xgb_params)
        params['num_class'] = len(unique_labels)

        clf = XGBClassifier(**params)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        clf.fit(X, y_encoded, **fit_kwargs)

        # Store classifier and label mapping
        self.classifiers[param_name] = clf
        # Store reverse mapping: index -> original label value
        self.label_mappings[param_name] = {i: v for v, i in label_to_idx.items()}

        # Calculate training accuracy
        y_pred = clf.predict(X)
        accuracy = np.mean(y_pred == y_encoded)

        return accuracy

    def set_constant(self, param_name: str, value):
        """
        Register a constant predictor (Tier 2): always predict this value.

        Args:
            param_name: Parameter name
            value: The constant value to always predict
        """
        self.constant_params[param_name] = value

    def set_bool_flip(self, param_name: str):
        """
        Register a bool flip predictor (Tier 3): always flip to the other boolean value.

        Args:
            param_name: Parameter name (must be bool type)
        """
        self.bool_flip_params.add(param_name)

    def _convert_value(self, value, param_type):
        """Convert string value back to original type based on parameter definition."""
        if isinstance(value, str):
            if param_type == 'int':
                return int(value)
            elif param_type == 'real':
                return float(value)
            elif param_type == 'bool':
                # Handle various string representations
                if value.lower() in ('true', '1', 'yes'):
                    return True
                elif value.lower() in ('false', '0', 'no'):
                    return False
            # char type stays as string
        return value

    def predict(self, features: np.ndarray, direction_vector: np.ndarray) -> Dict[str, any]:
        """
        Predict parameter values for ALL 30 parameters.

        Three-tier strategy (per parameter):
        1. XGBoost classifier (if trained)
        2. Constant value (if all training labels were the same)
        3. Bool flip (if boolean parameter)
        4. Default (no training data)

        Args:
            features: 20-dimensional feature vector
            direction_vector: 30-dimensional direction vector

        Returns:
            Dict mapping ALL parameter names to predicted values.
        """
        if features.shape[0] != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim}, got {features.shape[0]}")
        if direction_vector.shape[0] != self.n_params:
            raise ValueError(f"Expected direction dimension {self.n_params}, got {direction_vector.shape[0]}")

        # Build default config as base
        predictions = {p['name']: p['default'] for p in self.param_config['parameters']}

        # Concatenate features and direction vector
        X = np.concatenate([features, direction_vector]).reshape(1, -1)

        # Predict for each parameter
        for param_def in self.param_config['parameters']:
            param_name = param_def['name']
            candidates = param_def['candidates']
            default = param_def['default']

            # Tier 1: XGBoost classifier
            if param_name in self.classifiers:
                clf = self.classifiers[param_name]
                pred_idx = clf.predict(X)[0]
                if param_name in self.label_mappings and pred_idx in self.label_mappings[param_name]:
                    raw_value = self.label_mappings[param_name][pred_idx]
                    predictions[param_name] = self._convert_value(raw_value, param_def['type'])

            # Tier 2: Constant predictor
            elif param_name in self.constant_params:
                raw_value = self.constant_params[param_name]
                predictions[param_name] = self._convert_value(raw_value, param_def['type'])

            # Tier 3: Bool flip
            elif param_name in self.bool_flip_params:
                for c in candidates:
                    if c != default:
                        predictions[param_name] = c
                        break

            # Tier 4: Default (no training data) — already set above

        return predictions

    def predict_batch(
        self,
        features_batch: np.ndarray,
        direction_batch: np.ndarray,
    ) -> List[Dict[str, any]]:
        """
        Predict parameter values for multiple instances.

        Args:
            features_batch: Batch of features, shape (n_samples, 20)
            direction_batch: Batch of direction vectors, shape (n_samples, 30)

        Returns:
            List of prediction dicts
        """
        n_samples = features_batch.shape[0]
        if direction_batch.shape[0] != n_samples:
            raise ValueError("Batch sizes must match")

        all_predictions = []
        for i in range(n_samples):
            pred = self.predict(features_batch[i], direction_batch[i])
            all_predictions.append(pred)

        return all_predictions

    def save(self, save_dir: str):
        """
        Save all predictors to disk.

        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for param_name, clf in self.classifiers.items():
            safe_name = param_name.replace('/', '_')
            model_path = save_path / f"{safe_name}.json"
            clf.save_model(str(model_path))

        # Save metadata (includes all three tiers + label mappings)
        meta = {
            'n_params': self.n_params,
            'input_dim': self.input_dim,
            'trained_params': list(self.classifiers.keys()),
            'label_mappings': {k: {str(idx): _serialize_value(v) for idx, v in mapping.items()}
                               for k, mapping in self.label_mappings.items()},
            'constant_params': {k: _serialize_value(v) for k, v in self.constant_params.items()},
            'bool_flip_params': list(self.bool_flip_params),
        }
        meta_path = save_path / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    def load(self, load_dir: str):
        """
        Load predictors from disk.

        Args:
            load_dir: Directory containing saved models
        """
        load_path = Path(load_dir)

        # Load metadata
        meta_path = load_path / "metadata.json"
        if not meta_path.exists():
            print(f"WARNING: No metadata found at {meta_path}")
            return

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # Load XGBoost classifiers
        for param_name in meta.get('trained_params', []):
            safe_name = param_name.replace('/', '_')
            model_path = load_path / f"{safe_name}.json"

            if not model_path.exists():
                print(f"WARNING: Model file not found: {model_path}")
                continue

            clf = XGBClassifier()
            clf.load_model(str(model_path))
            self.classifiers[param_name] = clf

        # Load label mappings
        self.label_mappings = {}
        for param_name, mapping in meta.get('label_mappings', {}).items():
            self.label_mappings[param_name] = {int(idx): v for idx, v in mapping.items()}

        # Load constant predictors
        self.constant_params = {}
        for param_name, value in meta.get('constant_params', {}).items():
            # Convert value back to correct type based on parameter definition
            try:
                param_def = self._get_param_def(param_name)
                param_type = param_def['type']
                if param_type == 'int':
                    value = int(value)
                elif param_type == 'real':
                    value = float(value)
                elif param_type == 'bool':
                    value = bool(value)
            except (ValueError, KeyError):
                pass  # Keep original value if conversion fails
            self.constant_params[param_name] = value

        # Load bool flip predictors
        self.bool_flip_params = set(meta.get('bool_flip_params', []))

    def _get_param_def(self, param_name: str) -> Dict:
        """Get parameter definition by name."""
        for p in self.param_config['parameters']:
            if p['name'] == param_name:
                return p
        raise ValueError(f"Parameter {param_name} not found in config")

    def get_input_dim(self) -> int:
        """Get input dimension (50: 20 features + 30 directions)."""
        return self.input_dim

    def get_param_names(self) -> List[str]:
        """Get list of parameter names."""
        return [p['name'] for p in self.param_config['parameters']]

    def get_trained_params(self) -> List[str]:
        """Get list of parameter names that have any prediction strategy."""
        all_params = set(self.classifiers.keys())
        all_params.update(self.constant_params.keys())
        all_params.update(self.bool_flip_params)
        return sorted(all_params)

    def get_predictor_type(self, param_name: str) -> str:
        """Get the prediction strategy type for a parameter."""
        if param_name in self.classifiers:
            return "xgboost"
        elif param_name in self.constant_params:
            return "constant"
        elif param_name in self.bool_flip_params:
            return "bool_flip"
        else:
            return "default"


def _serialize_value(v):
    """Serialize a parameter value for JSON storage."""
    if isinstance(v, (np.integer,)):
        return int(v)
    elif isinstance(v, (np.floating,)):
        return float(v)
    elif isinstance(v, np.bool_):
        return bool(v)
    return v
