from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import httpx

# from pathlib import Path
# import sys
# PRJ_ROOT = Path(__file__).resolve().parent.parent.parent
# sys.path.insert(0, str(PRJ_ROOT))
from common.llm_connection.LLMBase import (
    LLMConnection,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
)
from common.llm_connection.token_counter import HuggingFaceTokenizer

logger = logging.getLogger(__name__)


def _to_chunk(line: str) -> Optional[LLMStreamChunk]:
    """
    Parse a single SSE line from OpenAI-compatible streaming response.

    Notes:
    - Token count here is only an estimate based on delta text.
    - Designed for performance testing, not billing-accurate accounting.
    """
    if not line:
        return None

    line = line.strip()
    if not line.startswith("data:"):
        return None

    raw = line[len("data:") :].strip()
    if raw == "[DONE]":
        return LLMStreamChunk(
            text="",
            num_tokens=0,
            is_finished=True,
            finish_reason="stop",
        )

    try:
        ev = json.loads(raw)
        choice = ev["choices"][0]
        delta = choice.get("delta", {})
        text = delta.get("content", "")
        finish_reason = choice.get("finish_reason")

        return LLMStreamChunk(
            text=text,
            num_tokens=0,  # filled later (estimate)
            is_finished=finish_reason is not None,
            finish_reason=finish_reason,
        )
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error("Failed to parse SSE line: %r (%s)", line, e)
        return None


@dataclass
class OpenAIConn(LLMConnection):
    """
    OpenAI-compatible LLM connection, intended for:
    - performance benchmarking
    - streaming latency measurement
    - basic accuracy testing

    Assumes /v1/chat/completions API.
    """

    base_url: str
    tokenizer: HuggingFaceTokenizer = field(repr=False)
    api_key: str = ""
    model: str = "default"

    timeout: float = 3600.0  # connect + read + write + pool

    _client: httpx.Client = field(init=False, repr=False)
    _aclient: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        limits = httpx.Limits(
            max_keepalive_connections=None,
            max_connections=None,
            keepalive_expiry=None,
        )

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            limits=limits,
        )
        self._aclient = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            limits=limits,
        )

    # ---------------- internal helpers ----------------

    def _make_body(self, req: LLMRequest) -> Dict[str, Any]:
        if req.messages:
            messages = [
                {"role": m["role"], "content": m["content"]} for m in req.messages
            ]
        elif req.num_tokens:
            logger.warning(
                "LLMRequest has no messages, using synthetic tokens for warmup"
            )
            messages = [
                {
                    "role": "user",
                    "content": self.tokenizer.get_some_tokens(req.num_tokens or 256),
                }
            ]
        else:
            raise ValueError("Either 'messages' or 'num_tokens' must be provided")

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": False,
        }

        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
            if req.ignore_eos is not None:
                body["ignore_eos"] = req.ignore_eos

        return body

    def _wrap_exception(self, e: Exception) -> Exception:
        if isinstance(e, httpx.HTTPStatusError):
            try:
                detail = e.response.json()
                msg = detail.get("error", {}).get("message", e.response.text)
            except Exception:
                msg = e.response.text
            raise RuntimeError(f"LLM API Error {e.response.status_code}: {msg}") from e

        if isinstance(e, httpx.RequestError):
            raise RuntimeError(
                f"LLM Network Error: {type(e).__name__} at {e.request.url}"
            ) from e

        raise e

    # ---------------- sync ----------------

    def chat(self, req: LLMRequest, **kwargs) -> LLMResponse:
        body = self._make_body(req)
        try:
            r = self._client.post("/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()

            # print(data)

            choice = data["choices"][0]
            text = choice["message"]["content"]

            return LLMResponse(
                text=text,
                finish_reason=choice.get("finish_reason"),
                total_tokens=self.tokenizer.count_tokens(text),
            )
        except Exception as e:
            raise self._wrap_exception(e)

    def stream_chat(self, req: LLMRequest, **kwargs) -> Iterator[LLMStreamChunk]:
        body = self._make_body(req)
        body["stream"] = True

        try:
            with self._client.stream(
                "POST",
                "/chat/completions",
                json=body,
            ) as resp:
                resp.raise_for_status()
                res = ""
                for line in resp.iter_lines():
                    chunk = _to_chunk(line)
                    if not chunk:
                        continue
                    if not chunk.is_finished:
                        # estimated token count for throughput only
                        chunk.num_tokens = self.tokenizer.count_tokens(chunk.text)
                    yield chunk
        except Exception as e:
            raise self._wrap_exception(e)

    # ---------------- async ----------------

    async def achat(self, req: LLMRequest, **kwargs) -> LLMResponse:
        body = self._make_body(req)
        try:
            r = await self._aclient.post("/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()

            choice = data["choices"][0]
            text = choice["message"]["content"]

            return LLMResponse(
                text=text,
                finish_reason=choice.get("finish_reason"),
                total_tokens=self.tokenizer.count_tokens(text),
            )
        except Exception as e:
            raise self._wrap_exception(e)

    async def astream_chat(
        self,
        req: LLMRequest,
        **kwargs,
    ) -> AsyncIterator[LLMStreamChunk]:
        body = self._make_body(req)
        body["stream"] = True

        try:
            async with self._aclient.stream(
                "POST",
                "/chat/completions",
                json=body,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    chunk = _to_chunk(line)
                    if not chunk:
                        continue
                    if not chunk.is_finished:
                        chunk.num_tokens = self.tokenizer.count_tokens(chunk.text)
                    yield chunk
        except Exception as e:
            raise self._wrap_exception(e)

    # ---------------- lifecycle ----------------

    def close(self):
        self._client.close()

    async def aclose(self):
        await self._aclient.aclose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# Example usage (for local testing)
if __name__ == "__main__":
    tok = HuggingFaceTokenizer("D:/Models/Qwen3-32B")
    import os

    conn = OpenAIConn(
        base_url="https://api.siliconflow.cn/v1",
        api_key=os.getenv("SILICON_API_KEY") or "",
        model="Qwen/Qwen2-7B-Instruct",
        tokenizer=tok,
    )

    # 1. Synchronous non-streaming test
    print("Test synchronous non-streaming")
    req = LLMRequest(
        messages=[{"role": "user", "content": "Hello, introduce yourself"}],
        max_tokens=1024,
    )
    print(conn.chat(req))  #

    # 2. Synchronous streaming test
    print("Test synchronous streaming")
    req = LLMRequest(num_tokens=10)
    res = ""
    stream_num = 0
    for c in conn.stream_chat(req):
        # print(c.text, end="", flush=True)
        # print(c.num_tokens, c.is_finished, c.finish_reason)
        if c.text:
            res += c.text
        stream_num += c.num_tokens
    print(f"Non Stream: {tok.count_tokens(res)}")  # Recommended for use
    print(f"Stream: {stream_num}")  # result is consistent with that Non Stream
