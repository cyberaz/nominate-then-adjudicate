# src/llm/__init__.py
# LLM integration: API client, direction recommender, and skill manager
#
# Default LLM provider: Qwen 3.7 Max via Anthropic-compatible API
# Set environment variables:
#   ANTHROPIC_AUTH_TOKEN = your API token
#   ANTHROPIC_BASE_URL   = https://api.example.invalid  (default)
#   ANTHROPIC_MODEL      = REDACTED_MODEL_ALIAS  (default)
from .client import LLMClient
from .direction_recommender import DirectionRecommender
from .skill_manager import SkillManager
