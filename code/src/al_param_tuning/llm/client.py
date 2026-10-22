"""
LLM API Client wrapper

Supports multiple LLM API providers through a unified interface:
1. OpenAI-compatible APIs (OpenAI, Azure, etc.)
2. Anthropic-compatible APIs (Anthropic, custom endpoints like Qwen)

Primary use case in this project:
- Semantic question answering (Phase A3): structured questions for embeddings
- Parameter space compression (Phase A1): LLM-guided parameter selection

Default configuration uses Qwen 3.7 Max via Anthropic-compatible endpoint:
    ANTHROPIC_AUTH_TOKEN = your token
    ANTHROPIC_BASE_URL   = https://api.example.invalid
    ANTHROPIC_MODEL      = REDACTED_MODEL_ALIAS

Usage:
    # Default: Qwen 3.7 Max via Anthropic-compatible API
    client = LLMClient()
    response = client.ask("Your prompt", system_prompt="You are an expert...")

    # Custom provider
    client = LLMClient(provider="openai", model="gpt-4", api_key="sk-...")
"""

import os
import json
import re
from typing import Optional, Dict, Any, List


# ── Default configuration for Qwen 3.7 Max ────────────────────────────
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "REDACTED_MODEL_ALIAS"
DEFAULT_BASE_URL = "https://api.example.invalid"
DEFAULT_API_KEY_ENV = "ANTHROPIC_AUTH_TOKEN"


class LLMClient:
    """
    Unified LLM API client supporting multiple providers.

    Supports:
    - Anthropic-compatible APIs (including custom endpoints like Qwen)
    - OpenAI-compatible APIs (including Azure, local servers)

    The Anthropic provider supports custom `base_url` for self-hosted
    or proxy endpoints (e.g., Qwen via Anthropic-compatible gateway).
    """

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        """
        Initialize LLM client.

        Args:
            provider: "anthropic" or "openai"
            model: Model name (default: "REDACTED_MODEL_ALIAS")
            api_key: API key (defaults to ANTHROPIC_AUTH_TOKEN or OPENAI_API_KEY env var)
            base_url: Custom API base URL (for Anthropic-compatible endpoints)
                     Defaults to https://api.example.invalid for anthropic provider
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (low for deterministic responses)
            timeout: Per-request timeout in seconds. The SDK default is 600s, which lets
                     one stalled request hold up an entire concurrent batch.
            max_retries: SDK-level retries for transient/connection failures
        """
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        # Export-only adapter: explicit shell configuration; no local resource files.
        if provider == "openai":
            if api_key is None or api_key in {"OPENAI_API_KEY", "DASHSCOPE_API_KEY"}:
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
            if base_url is None:
                base_url = os.getenv("OPENAI_BASE_URL")
        elif provider == "anthropic" and base_url is None:
            base_url = os.getenv("ANTHROPIC_BASE_URL")
        if not base_url:
            raise ValueError("Set an explicit API base URL in the shell before online execution.")

        # Resolve API key
        if api_key in {"OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"}:
            api_key = os.getenv(api_key)

        if api_key is None:
            if provider == "anthropic":
                api_key = os.getenv(DEFAULT_API_KEY_ENV) or os.getenv("ANTHROPIC_API_KEY")
            elif provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")

        if api_key is None:
            env_hint = DEFAULT_API_KEY_ENV if provider == "anthropic" else "OPENAI_API_KEY"
            raise ValueError(
                f"API key not provided. Set {env_hint} env var or pass api_key parameter."
            )

        self.api_key = api_key

        # Resolve base_url
        if base_url is None and provider == "anthropic":
            base_url = os.getenv("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)
        self.base_url = base_url

        # Lazy-load API clients
        self._client = None

    def _get_anthropic_client(self):
        """Lazy-load Anthropic client (works for Anthropic-compatible endpoints)."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Run: pip install anthropic"
                )

            kwargs = {"api_key": self.api_key,
                      "timeout": self.timeout,
                      "max_retries": self.max_retries}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _get_openai_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                )

            kwargs = {"api_key": self.api_key,
                      "timeout": self.timeout,
                      "max_retries": self.max_retries}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = OpenAI(**kwargs)
        return self._client

    def batch_ask(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        max_tokens_override: Optional[int] = None,
        max_concurrent: int = 8,
    ) -> List[str]:
        """
        Send multiple prompts to the LLM in parallel with rate limiting.

        Args:
            prompts: List of prompts to send
            system_prompt: System prompt (optional)
            json_mode: If True, hint the model to return JSON
            max_tokens_override: Override default max_tokens for all calls
            max_concurrent: Maximum number of concurrent API calls (default: 4)

        Returns:
            List of response texts in the same order as prompts
        """
        import concurrent.futures
        import threading
        from tqdm import tqdm

        results = [None] * len(prompts)
        lock = threading.Lock()
        semaphore = threading.Semaphore(max_concurrent)
        errors = []

        def ask_single(idx, prompt):
            semaphore.acquire()
            try:
                result = self.ask(prompt, system_prompt, json_mode, max_tokens_override)
                with lock:
                    results[idx] = result
            except Exception as e:
                with lock:
                    errors.append((idx, str(e)))
            finally:
                semaphore.release()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [
                executor.submit(ask_single, idx, prompt)
                for idx, prompt in enumerate(prompts)
            ]

            # Use as_completed with tqdm for progress
            for _ in tqdm(concurrent.futures.as_completed(futures),
                         total=len(futures),
                         desc=f"    LLM API calls ({max_concurrent} concurrent)"):
                pass

        if errors:
            print(f"\n    Warning: {len(errors)} LLM calls failed:")
            for idx, err in errors[:3]:
                print(f"      [{idx}] {err[:100]}")

        return results

    def ask(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        max_tokens_override: Optional[int] = None,
    ) -> str:
        """
        Send a prompt to the LLM and return the response.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            json_mode: If True, hint the model to return JSON
                      (Anthropic: appended to prompt; OpenAI: response_format)
            max_tokens_override: Override default max_tokens for this call only

        Returns:
            LLM response text
        """
        saved = self.max_tokens
        if max_tokens_override is not None:
            self.max_tokens = max_tokens_override
        try:
            if self.provider == "anthropic":
                result = self._ask_anthropic(prompt, system_prompt, json_mode)
            elif self.provider == "openai":
                result = self._ask_openai(prompt, system_prompt, json_mode)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        finally:
            self.max_tokens = saved
        return result

    def _ask_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        json_mode: bool,
        max_retries: int = 3,
    ) -> str:
        """Call Anthropic-compatible API with retry logic."""
        import time

        client = self._get_anthropic_client()

        # For Anthropic-compatible APIs that may not support json_mode natively,
        # append a hint to the prompt
        actual_prompt = prompt
        if json_mode:
            actual_prompt = prompt + "\n\n(Please respond with valid JSON only.)"

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": actual_prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        last_error = None
        for attempt in range(max_retries):
            try:
                response = client.messages.create(**kwargs)

                # Check for empty response
                if response.content is None:
                    raise ValueError("API returned empty content (None)")

                # Extract text from response content blocks
                # Qwen 3.7 Max may return ThinkingBlock (reasoning) + TextBlock (actual response)
                text_parts = []
                for block in response.content:
                    if hasattr(block, 'text'):
                        text_parts.append(block.text)
                    elif hasattr(block, 'thinking'):
                        # Skip thinking blocks, they're internal reasoning
                        pass

                if not text_parts:
                    raise ValueError("No text content found in LLM response")

                return "\n".join(text_parts)

            except (ValueError, Exception) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    print(f"  [LLM Retry] Attempt {attempt + 1}/{max_retries} failed: {e}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    break

        raise ValueError(f"LLM API failed after {max_retries} attempts: {last_error}")

    def _ask_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        json_mode: bool,
    ) -> str:
        """Call OpenAI-compatible API."""
        client = self._get_openai_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def ask_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a prompt and parse JSON response.

        Handles common LLM quirks:
        - JSON wrapped in markdown code blocks
        - Extra whitespace or explanatory text around JSON

        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: System prompt

        Returns:
            Parsed JSON dict
        """
        response = self.ask(prompt, system_prompt, json_mode=True)
        return self._parse_json_response(response)

    @staticmethod
    def _parse_json_response(response_text: str) -> Dict[str, Any]:
        """
        Robustly parse JSON from LLM response.

        Handles:
        - Clean JSON
        - JSON in markdown code blocks (```json ... ```)
        - JSON with leading/trailing text
        """
        # 1. Try direct parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # 2. Try extracting from markdown code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Try finding first { ... } block
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Failed to parse JSON from LLM response:\n{response_text[:500]}"
        )

    def test_connection(self) -> bool:
        """
        Test API connection with a simple ping.

        Returns:
            True if connection successful
        """
        try:
            self.ask("Reply with exactly: OK", max_tokens_override=10)
            return True
        except Exception:
            return False
