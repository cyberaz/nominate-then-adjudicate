"""
Inference Pipeline (Multi-Parameter Version)

Uses trained models to recommend parameter configurations for new MIP instances.
Two-step process:
  Step 1: LLM recommends 30-dim direction vector
  Step 2: XGBoost predicts all 30 parameter values
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml

from ..features.instance_features import InstanceFeatureExtractor
from ..llm import LLMClient, DirectionRecommender
from ..model.param_predictor import ParameterPredictor


class InferencePipeline:
    """
    Inference pipeline for multi-parameter recommendation.

    Workflow:
    1. Extract 20-dim features from MIP instance
    2. LLM recommends 30-dim direction vector — reads skill.md
    3. XGBoost predicts all 30 parameter values from [features, direction]
    """

    def __init__(
        self,
        config_path: str = "config/experiment.yaml",
        model_path: Optional[str] = None,
    ):
        """
        Args:
            config_path: Path to experiment configuration
            model_path: Override path to trained model (from config if None)
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Paths
        self.param_config_path = self.config['paths']['param_config']
        self.skill_path = str(Path(self.config['paths']['skill_dir']) / "skill.md")
        model_path = model_path or self.config['training']['model_save_path']

        # Initialize components
        self.feature_extractor = InstanceFeatureExtractor()

        # LLM
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
            skill_path=self.skill_path,
        )

        # Load trained predictor
        self.predictor = ParameterPredictor(param_config_path=self.param_config_path)
        if Path(model_path).exists():
            self.predictor.load(model_path)
            print(f"Loaded trained model from {model_path}")
        else:
            print(f"WARNING: No trained model found at {model_path}. "
                  "Predictor will return default values.")

    def recommend(self, model) -> Dict:
        """
        Recommend parameter configuration for a MIP instance.

        Args:
            model: PySCIPOpt Model object

        Returns:
            Dict mapping parameter names to recommended values
        """
        # Step 1: Extract features
        features = self.feature_extractor.extract(model)

        # Step 2: Get LLM 30-dim direction vector
        direction = self.direction_recommender.recommend(features, model)

        # Step 3: XGBoost predicts all 30 params
        config = self.predictor.predict(features, direction)

        return config

    def recommend_with_details(self, model) -> Dict:
        """
        Recommend configuration with intermediate results.

        Returns:
            Dict with keys:
                - config: recommended parameter values
                - features: 20-dim feature vector
                - direction: 30-dim direction vector
        """
        # Extract features
        features = self.feature_extractor.extract(model)

        # Get LLM direction
        direction = self.direction_recommender.recommend(features, model)

        # Predict
        config = self.predictor.predict(features, direction)

        return {
            'config': config,
            'features': features.tolist(),
            'direction': direction.tolist(),
        }

    def recommend_batch(self, models: list) -> list:
        """
        Recommend configurations for multiple instances.

        Args:
            models: List of PySCIPOpt Model objects

        Returns:
            List of config dicts
        """
        configs = []
        for model in models:
            config = self.recommend(model)
            configs.append(config)

        return configs
