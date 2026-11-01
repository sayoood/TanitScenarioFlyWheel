"""Local LLM client: Qwen3.8-27B (NVFP4) served by vLLM on thor3, OpenAI-compatible API.

Structured outputs use vLLM/xgrammar JSON-schema constrained decoding, so agent outputs are
schema-valid by construction. Every call is recorded (latency, tokens) for the KPI store.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from openai import OpenAI

BASE_URL = os.environ.get("TSF_LLM_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("TSF_LLM_MODEL", "qwen3.8-27b")


@dataclass
class CallRecord:
    role: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    ok: bool
    error: str = ""

    @property
    def tok_per_s(self) -> float:
        return round(self.completion_tokens / self.latency_s, 1) if self.latency_s else 0.0


@dataclass
class LLM:
    base_url: str = BASE_URL
    model: str = MODEL
    calls: list[CallRecord] = field(default_factory=list)

    def __post_init__(self):
        self.client = OpenAI(base_url=self.base_url, api_key=os.environ.get("TSF_LLM_KEY", "local"), timeout=900)

    def chat(self, messages: list[dict], schema: dict | None = None, role: str = "agent", thinking: bool = False,
             temperature: float = 0.3, max_tokens: int = 8192, retries: int = 1) -> str | dict:
        extra: dict = {"chat_template_kwargs": {"enable_thinking": thinking}}
        if schema is not None:
            # compact JSON grammar: free whitespace lets the model fall into endless newline loops
            extra["structured_outputs"] = {"json": schema, "disable_any_whitespace": True}
            extra["repetition_penalty"] = 1.03
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature,
                                                        top_p=0.9, max_tokens=max_tokens, extra_body=extra)
            except Exception as e:
                self.calls.append(CallRecord(role, time.perf_counter() - t0, 0, 0, False, str(e)[:300]))
                raise
            u = r.usage
            text = r.choices[0].message.content or ""
            finish = r.choices[0].finish_reason
            rec = CallRecord(role, time.perf_counter() - t0, u.prompt_tokens if u else 0, u.completion_tokens if u else 0,
                             True)
            self.calls.append(rec)
            if schema is None:
                return text
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                rec.ok, rec.error = False, f"invalid JSON (finish={finish}): {e}"[:300]
                if attempt == retries:
                    raise
                temperature = min(1.0, temperature + 0.3)
        raise RuntimeError("unreachable")

    def stats(self) -> dict:
        ok = [c for c in self.calls if c.ok]
        lat = sum(c.latency_s for c in ok)
        comp = sum(c.completion_tokens for c in ok)
        return {"calls": len(self.calls), "failed": len(self.calls) - len(ok), "latency_s": round(lat, 1),
                "prompt_tokens": sum(c.prompt_tokens for c in ok), "completion_tokens": comp,
                "tok_per_s": round(comp / lat, 1) if lat else 0.0}
