#!/usr/bin/env python3
"""Quick test of LLM API availability for SignalForge.

Usage:
    python scripts/test_llm_api.py
"""

import os
import sys
import time


def _get_genai_client():
    """Try new SDK first, fall back to legacy."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None, None

    # Try new google-genai SDK first
    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=api_key)
        return "new", client
    except ImportError:
        pass

    # Fall back to legacy SDK
    try:
        import google.generativeai as genai_old
        genai_old.configure(api_key=api_key)
        return "legacy", genai_old
    except ImportError:
        return None, None


def test_gemini() -> bool:
    """Test Gemini API connectivity and generation."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("  SKIP  gemini  GEMINI_API_KEY not set")
        return False

    sdk_type, client = _get_genai_client()
    if sdk_type is None:
        print("  FAIL  gemini  No SDK installed (pip install google-genai)")
        return False

    print(f"  INFO  gemini  Using {sdk_type} SDK")

    try:
        # List models
        if sdk_type == "new":
            models = [m.name for m in client.models.list()]
            gen_models = [m for m in models if "gemini" in m]
        else:
            models = [m.name for m in client.list_models() if "generateContent" in (m.supported_generation_methods or [])]
            gen_models = models

        print(f"  INFO  gemini  {len(gen_models)} models available")
        for m in gen_models[:5]:
            print(f"         - {m}")
        if len(gen_models) > 5:
            print(f"         ... and {len(gen_models) - 5} more")

        # Test generation
        t0 = time.time()
        if sdk_type == "new":
            from google.genai import types as genai_types
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Say 'SignalForge API test OK' and nothing else.",
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=20,
                ),
            )
            text = (response.text or "").strip()
        else:
            model = client.GenerativeModel(
                "gemini-2.5-flash",
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                },
            )
            response = model.generate_content(
                "Say 'SignalForge API test OK' and nothing else.",
                generation_config=client.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=20,
                ),
            )
            text = (response.text or "").strip()

        elapsed = time.time() - t0
        print(f"  OK    gemini  model=gemini-2.5-flash  response='{text}'  latency={elapsed:.2f}s")
        return True

    except Exception as e:
        print(f"  FAIL  gemini  {e}")
        return False


def test_gemini_trading_prompt() -> bool:
    """Test Gemini with a realistic trading analysis prompt."""
    sdk_type, client = _get_genai_client()
    if sdk_type is None:
        print("  SKIP  gemini-trading  No SDK available")
        return False

    prompt = """You are a Technical Analyst for a trading system. Given this market data:
- Current Price: $250.00
- 5-Day Return: +2.1%
- RSI(14): 58.3
- MACD Histogram: +0.45

Provide your analysis in this exact format:
SIGNAL: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
CONFIDENCE: [0.0 to 1.0]
ANALYSIS: [1 sentence]"""

    try:
        t0 = time.time()
        if sdk_type == "new":
            from google.genai import types as genai_types
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=200,
                ),
            )
            text = (response.text or "").strip()
        else:
            model = client.GenerativeModel(
                "gemini-2.5-flash",
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                },
            )
            response = model.generate_content(
                prompt,
                generation_config=client.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=200,
                ),
            )
            text = (response.text or "").strip()

        elapsed = time.time() - t0

        # Parse signal
        signal = "UNKNOWN"
        for line in text.split("\n"):
            if line.strip().upper().startswith("SIGNAL:"):
                signal = line.split(":", 1)[1].strip()
                break

        print(f"  OK    gemini-trading  signal={signal}  latency={elapsed:.2f}s")
        print(f"         Response: {text[:200]}")
        return True

    except Exception as e:
        print(f"  FAIL  gemini-trading  {e}")
        return False


def test_env_vars() -> None:
    """Check relevant environment variables."""
    keys = [
        ("GEMINI_API_KEY", "Google Gemini"),
        ("DEFAULT_GEMINI_KEY", "Gemini (fallback)"),
        ("ANTHROPIC_API_KEY", "Anthropic Claude"),
        ("OPENAI_API_KEY", "OpenAI"),
    ]
    print("\n=== Environment Variables ===")
    for var, label in keys:
        val = os.environ.get(var, "")
        if val:
            masked = val[:8] + "..." + val[-4:] if len(val) > 12 else "***"
            print(f"  SET   {var}  ({label})  {masked}")
        else:
            print(f"  --    {var}  ({label})  not set")


def main() -> None:
    print("=== SignalForge LLM API Test ===\n")

    test_env_vars()

    print("\n=== Gemini API Test ===")
    ok1 = test_gemini()

    if ok1:
        print("\n=== Gemini Trading Prompt Test ===")
        test_gemini_trading_prompt()

    print("\n=== Summary ===")
    if ok1:
        print("  Gemini API: READY")
        print("  Agents engine will use: gemini-2.5-flash")
    else:
        print("  Gemini API: NOT AVAILABLE")
        print("  Agents engine will use: rule-based fallback")

    print("\n=== Install Commands ===")
    sdk_type, _ = _get_genai_client()
    if sdk_type != "new":
        print("  Recommended: pip install google-genai")
    else:
        print("  google-genai SDK: installed")


if __name__ == "__main__":
    main()
