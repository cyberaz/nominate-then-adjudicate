# src/features/__init__.py
# Feature extraction: 20-dimensional instance features + rich text descriptions
from .instance_features import InstanceFeatureExtractor
from .instance_descriptor import InstanceDescriptor

__all__ = ["InstanceFeatureExtractor", "InstanceDescriptor"]
