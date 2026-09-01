"""OpenAI client wrapper: structured-output calls, retry/backoff, brain-model
fallback, and per-call usage/cost accounting.

Verified against the live account (scripts/verify_openai.py,
scripts/probe_openai_api.py) on 2026-08-31: the Responses API's actual usage
object shape is
    {input_tokens, input_tokens_details: {cached_tokens, cache_write_tokens},
     output_tokens, output_tokens_details: {reasoning_tokens}, total_tokens}
-- this module reads exactly those fields, not an assumed/pre-cutoff shape.
"""
import hashlib
import json
import os
import random
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_json(rel_path):
    with open(os.path.join(ROOT, rel_path), encoding="utf-8") as fh:
        return json.load(fh)


class LLMConfigError(Exception):
    pass


class LLMCallError(Exception):
    pass


class LLMClient:
    ROLES = ("brain", "repair", "fallback")

    def __init__(self, models_path="config/models.json", pricing_path="config/pricing.json", client=None):
        self.models = _load_json(models_path)
        self.pricing = _load_json(pricing_path)
        if not self.models.get("verified"):
            raise LLMConfigError(
                "config/models.json is not marked verified -- run scripts/verify_openai.py "
                "and scripts/probe_openai_api.py against the live account first."
            )
        if self.pricing.get("UNVERIFIED"):
            raise LLMConfigError("config/pricing.json is still UNVERIFIED -- confirm current prices first.")
        # Explicit timeout rather than relying on the SDK default: a stalled
        # connection should surface as a retryable APITimeoutError within a
        # bounded time, not hang the whole loop indefinitely. 480s is
        # generous for a high-reasoning-effort call on a large prompt while
        # still being well inside the sandbox's own node timeouts.
        self._client = client or OpenAI(timeout=480.0)
        self._consecutive_brain_failures = 0

    def _model_for(self, role):
        return self.models[role]

    def _effort_for(self, role):
        if role == "brain":
            return self.models.get("brain_reasoning_effort")
        if role in ("repair", "fallback"):
            return self.models.get("repair_reasoning_effort") if role == "repair" else self.models.get("brain_reasoning_effort")
        return None

    def _rate(self, model_id):
        rates = self.pricing["rates_usd_per_mtok"]
        if model_id not in rates:
            raise LLMConfigError(f"no pricing entry for model {model_id!r} in config/pricing.json")
        return rates[model_id]

    def _cost_usd(self, model_id, usage):
        rate = self._rate(model_id)
        details_in = usage.get("input_tokens_details") or {}
        details_out = usage.get("output_tokens_details") or {}
        cached = details_in.get("cached_tokens", 0)
        uncached_input = max(0, usage.get("input_tokens", 0) - cached)
        cost = (
            uncached_input / 1e6 * rate["input"]
            + cached / 1e6 * rate["cached_input"]
            + usage.get("output_tokens", 0) / 1e6 * rate["output"]
        )
        return cost

    def call(self, role, input_messages, schema, schema_name, max_retries=5):
        """Calls the model assigned to `role` with a strict JSON-schema
        response format. Returns (parsed_dict, usage_record). usage_record
        always has: role, model, input_tokens, cached_tokens, output_tokens,
        reasoning_tokens, total_tokens, cost_usd, latency_s, attempts,
        prompt_sha256, http_status (on failure), raw_usage.
        """
        if role not in self.ROLES:
            raise ValueError(f"unknown role {role!r}")

        model_id = self._model_for(role)
        effort = self._effort_for(role)
        prompt_sha256 = hashlib.sha256(json.dumps(input_messages, sort_keys=True).encode("utf-8")).hexdigest()

        kwargs = dict(
            model=model_id,
            input=input_messages,
            text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
        )
        if effort:
            kwargs["reasoning"] = {"effort": effort}

        last_exc = None
        t0 = time.time()
        print(f"[llm] -> {role} ({model_id}) call starting, prompt {prompt_sha256[:10]}...", flush=True)
        for attempt in range(1, max_retries + 1):
            try:
                t_attempt = time.time()
                resp = self._client.responses.create(**kwargs)
                print(f"[llm] <- {role} attempt {attempt} succeeded in {time.time()-t_attempt:.1f}s", flush=True)
                break
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                last_exc = e
                wait = min(30, (2 ** attempt) + random.uniform(0, 1))
                print(f"[llm] {role} attempt {attempt} failed after {time.time()-t_attempt:.1f}s "
                      f"({type(e).__name__}: {e}); retrying in {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            except APIStatusError as e:
                last_exc = e
                if 500 <= getattr(e, "status_code", 0) < 600:
                    wait = min(30, (2 ** attempt) + random.uniform(0, 1))
                    print(f"[llm] {role} attempt {attempt} got {e.status_code}; retrying in {wait:.1f}s", flush=True)
                    time.sleep(wait)
                    continue
                print(f"[llm] {role} attempt {attempt} non-retryable status {e.status_code}", flush=True)
                raise LLMCallError(f"{role} call failed with non-retryable status {e.status_code}: {e}") from e
        else:
            if role == "brain":
                self._consecutive_brain_failures += 1
            print(f"[llm] {role} call FAILED after {max_retries} attempts, {time.time()-t0:.1f}s total", flush=True)
            raise LLMCallError(f"{role} call failed after {max_retries} attempts: {last_exc}") from last_exc

        if role == "brain":
            self._consecutive_brain_failures = 0

        latency_s = time.time() - t0
        usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage or {})
        cached_tokens = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        reasoning_tokens = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        cost_usd = self._cost_usd(model_id, usage)

        try:
            parsed = json.loads(resp.output_text)
        except (json.JSONDecodeError, AttributeError) as e:
            raise LLMCallError(f"{role} call returned non-JSON output despite strict schema: {e}") from e

        usage_record = {
            "role": role,
            "model": model_id,
            "input_tokens": usage.get("input_tokens", 0),
            "cached_tokens": cached_tokens,
            "output_tokens": usage.get("output_tokens", 0),
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": usage.get("total_tokens", 0),
            "cost_usd": cost_usd,
            "latency_s": latency_s,
            "attempts": attempt,
            "prompt_sha256": prompt_sha256,
            "raw_usage": usage,
        }
        return parsed, usage_record

    def should_use_fallback(self, threshold=2):
        return self._consecutive_brain_failures >= threshold
