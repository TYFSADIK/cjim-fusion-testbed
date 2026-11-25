"""Talks to vLLM through its OpenAI-compatible API: chat with JSON-schema-constrained output
(vLLM structured outputs, so a reply always parses), and embeddings from a second vLLM instance
serving an embedding model.

  VLLM_URL        chat server, e.g. http://127.0.0.1:8000/v1
  VLLM_MODEL      the model name it serves
  VLLM_EMBED_URL  embedding server, e.g. http://127.0.0.1:8001/v1
  VLLM_EMBED_MODEL
"""
import json
import os
import time

import httpx

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.5-2b")
EMBED_URL = os.environ.get("VLLM_EMBED_URL", "http://127.0.0.1:8001/v1")
EMBED_MODEL = os.environ.get("VLLM_EMBED_MODEL", "qwen3-embedding")
_client = httpx.Client(timeout=180)
stats = {"chat_calls": 0, "chat_s": 0.0, "embed_calls": 0, "embed_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


def chat_json(system, user, schema, max_tokens=400, temperature=0.0):
    """One chat completion whose output is constrained to `schema`; returns the parsed object."""
    t0 = time.time()
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True}},
        # Qwen3.5 thinks before answering by default; extraction and summaries do not need it
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = _client.post(f"{URL}/chat/completions", json=body)
    r.raise_for_status()
    d = r.json()
    stats["chat_calls"] += 1
    stats["chat_s"] += time.time() - t0
    u = d.get("usage") or {}
    stats["prompt_tokens"] += u.get("prompt_tokens", 0)
    stats["completion_tokens"] += u.get("completion_tokens", 0)
    return json.loads(d["choices"][0]["message"]["content"])


def embed(texts):
    """Embeddings for a list of texts, L2-normalised by the server."""
    t0 = time.time()
    r = _client.post(f"{EMBED_URL}/embeddings", json={"model": EMBED_MODEL, "input": texts})
    r.raise_for_status()
    stats["embed_calls"] += 1
    stats["embed_s"] += time.time() - t0
    return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]


def ready(url=None):
    try:
        return httpx.get(f"{url or URL}/models", timeout=3).status_code == 200
    except Exception:
        return False
