"""Provider registry for the AI builder's OpenAI-compatible chat API.

Both AI-builder paths — the single-shot one in :mod:`nl_builder` and the agentic
tool-caller in :mod:`nl_agent` — only ever speak the OpenAI *chat-completions*
protocol: pick an action and fill a JSON schema, or call function tools. Any
service that exposes that protocol works, so the builder is open to any AI API,
not just Groq. This module maps a short provider name to its endpoint URL, the
environment variable its key comes from, and a sensible default model; it also
provides the single ``urllib`` POST both builders share.

Choosing an endpoint, in precedence order:

* an explicit ``base_url`` (the UI's "Custom" provider — point it at anything
  OpenAI-compatible, e.g. a local Ollama / LM Studio / vLLM server);
* a named ``provider`` from :data:`PROVIDERS`;
* otherwise :data:`DEFAULT_PROVIDER` (``groq`` — free and reliable for this).

Keys are never hardcoded or written to disk: they come from the per-request
``api_key`` (pasted in the UI) or the provider's environment variable.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

# Groq's endpoint stays the historical default so existing setups keep working.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# name -> endpoint config. ``model`` is the single-shot default, ``agent_model``
# the (stronger) default for the tool-calling worker. ``key_env`` is the env var
# the key falls back to. ``local`` marks servers that don't need a real key.
PROVIDERS: dict[str, dict] = {
    "groq": {
        "label": "Groq (free)",
        "url": GROQ_URL,
        "key_env": "GROQ_API_KEY",
        "model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        "agent_model": os.environ.get("GROQ_AGENT_MODEL", "llama-3.3-70b-versatile"),
        "json_object": True,
    },
    "openai": {
        "label": "OpenAI",
        "url": "https://api.openai.com/v1/chat/completions",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "agent_model": "gpt-4o",
        "json_object": True,
    },
    "openrouter": {
        "label": "OpenRouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "model": "openai/gpt-4o-mini",
        "agent_model": "openai/gpt-4o",
        "json_object": True,
    },
    "together": {
        "label": "Together AI",
        "url": "https://api.together.xyz/v1/chat/completions",
        "key_env": "TOGETHER_API_KEY",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "agent_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "json_object": True,
    },
    "deepseek": {
        "label": "DeepSeek",
        "url": "https://api.deepseek.com/v1/chat/completions",
        "key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "agent_model": "deepseek-chat",
        "json_object": True,
    },
    "mistral": {
        "label": "Mistral",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "key_env": "MISTRAL_API_KEY",
        "model": "mistral-small-latest",
        "agent_model": "mistral-large-latest",
        "json_object": True,
    },
    "anthropic": {
        # Anthropic's OpenAI-SDK compatibility endpoint. Tool calling works;
        # it does not accept response_format=json_object, so json_object is off
        # (the system prompt still asks for JSON-only, which it honours).
        "label": "Anthropic (Claude)",
        "url": "https://api.anthropic.com/v1/chat/completions",
        "key_env": "ANTHROPIC_API_KEY",
        "model": "claude-haiku-4-5",
        "agent_model": "claude-opus-4-8",
        "json_object": False,
    },
    "local": {
        # Ollama / LM Studio / vLLM and friends all serve /v1/chat/completions
        # locally and ignore the Authorization header.
        "label": "Local (Ollama / LM Studio)",
        "url": os.environ.get("LOCAL_AI_URL", "http://localhost:11434/v1/chat/completions"),
        "key_env": "OPENAI_API_KEY",
        "model": os.environ.get("LOCAL_AI_MODEL", "llama3.1"),
        "agent_model": os.environ.get("LOCAL_AI_MODEL", "llama3.1"),
        "json_object": True,
        "local": True,
    },
}

DEFAULT_PROVIDER = os.environ.get("AI_PROVIDER", "groq")


def provider_catalog() -> list[dict]:
    """Public, UI-facing summary of the providers (no secrets)."""
    catalog = []
    for name, cfg in PROVIDERS.items():
        catalog.append({
            "name": name,
            "label": cfg["label"],
            "model": cfg["model"],
            "agent_model": cfg["agent_model"],
            "key_env": cfg.get("key_env"),
            "local": bool(cfg.get("local")),
        })
    return catalog


def resolve(provider: str | None = None, base_url: str | None = None,
            model: str | None = None, api_key: str | None = None,
            agent: bool = False) -> tuple[str, str, str, bool]:
    """Resolve a request to ``(url, api_key, model, supports_json_object)``.

    ``base_url`` (when given) wins over the named provider's URL; ``model`` wins
    over the provider's default; ``api_key`` falls back to the provider's env var
    (or the generic ``AI_API_KEY``). Raises a clear error when no key is found and
    the endpoint isn't a local server.
    """
    name = (provider or DEFAULT_PROVIDER or "groq").strip().lower()
    cfg = PROVIDERS.get(name, {})
    url = (base_url or "").strip() or cfg.get("url") or GROQ_URL
    is_local = bool(cfg.get("local")) or url.startswith(("http://localhost",
                                                         "http://127.0.0.1"))

    key = (api_key or "").strip()
    if not key and cfg.get("key_env"):
        key = os.environ.get(cfg["key_env"], "")
    if not key:
        key = os.environ.get("AI_API_KEY", "")
    if not key:
        if is_local:
            key = "local"  # local servers ignore the Authorization header
        else:
            env = cfg.get("key_env", "AI_API_KEY")
            label = cfg.get("label", name)
            raise RuntimeError(
                f"No API key for {label}. Paste one in the AI builder box, "
                f"or set the {env} environment variable on the server."
            )

    chosen = (model or "").strip() or cfg.get("agent_model" if agent else "model")
    if not chosen:
        raise RuntimeError(
            "No model selected. Type a model name in the AI builder, or pick a "
            "provider with a built-in default."
        )
    supports_json = cfg.get("json_object", True)
    return url, key, chosen, supports_json


def chat(url: str, payload: dict, api_key: str, timeout: int = 60) -> dict:
    """One OpenAI-compatible chat-completions POST, over plain ``urllib``."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Some providers (Groq's Cloudflare front-end) reject urllib's
            # default "Python-urllib/3.x" User-Agent with a 403; send our own.
            "User-Agent": "vasp_auto/0.8 (+https://github.com/; python-urllib)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:  # bad key, rate limit, bad model name
        detail = exc.read().decode("utf-8", "ignore")[:300]
        raise RuntimeError(f"AI API error {exc.code}: {detail}") from None
