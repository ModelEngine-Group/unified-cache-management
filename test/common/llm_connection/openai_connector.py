# openai_conn.py
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx
from common.llm_connection.LLMBase import (
    LLMConnection,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
)
from common.llm_connection.token_counter import HuggingFaceTokenizer


# Utility: Convert OpenAI's streaming "delta" format into our LLMStreamChunk
def _to_chunk(line: str) -> Optional[LLMStreamChunk]:
    """
    Parses a single SSE (Server-Sent Events) line from OpenAI-compatible streaming responses.
    Example input line:
        data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}
    """
    if not line.startswith("data:"):
        return None
    raw = line[len("data:") :].strip()
    if raw == "[DONE]":
        return LLMStreamChunk(
            text="", num_tokens=0, is_finished=True, finish_reason="stop"
        )
    try:
        ev = json.loads(raw)
        delta = ev["choices"][0].get("delta", {})
        text = delta.get("content", "")
        finish_reason = ev["choices"][0].get("finish_reason")
        return LLMStreamChunk(
            text=text,
            num_tokens=len(
                text.encode()
            ),  # Placeholder; will be corrected later using tokenizer
            is_finished=finish_reason is not None,
            finish_reason=finish_reason,
        )
    except Exception:
        return None


# OpenAI-Compatible LLM Connector
@dataclass
class OpenAIConn(LLMConnection):
    # Fields without defaults come first
    base_url: str
    tokenizer: HuggingFaceTokenizer = field(repr=False)
    # Fields with defaults follow
    api_key: str = ""
    model: str = "default"
    timeout: float = 120.0

    # ---------- Internal: Build OpenAI-compatible request body ----------
    def _make_body(self, req: LLMRequest) -> Dict[str, Any]:
        """Converts an LLMRequest into an OpenAI-style JSON request body."""
        if req.messages:
            messages: List[Dict[str, str]] = [
                {"role": m["role"], "content": m["content"]} for m in req.messages
            ]
        elif req.num_tokens:
            # In num_tokens mode: generate dummy input with approximate token count
            messages = [
                {
                    "role": "user",
                    "content": self.tokenizer.get_some_tokens(req.num_tokens or 256),
                }
            ]
        else:
            raise TypeError("Either 'messages' or 'num_tokens' must be provided.")

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": req.temperature,
            "top_p": req.top_p,
        }
        if req.max_tokens:
            body["max_tokens"] = req.max_tokens
            # Note: 'ignore_eos' is non-standard; only supported by certain backends (e.g., vLLM)
            body["ignore_eos"] = req.ignore_eos or True
        return body

    def _headers(self) -> Dict[str, str]:
        """Constructs HTTP headers for the request."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ---------- Synchronous methods ----------
    def chat(self, req: LLMRequest, **kwargs) -> LLMResponse:
        """Performs a synchronous non-streaming chat completion."""
        body = self._make_body(req)
        body["stream"] = False
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            r = client.post("/chat/completions", json=body, headers=self._headers())
            r.raise_for_status()
            data = r.json()
        txt: str = data["choices"][0]["message"]["content"]
        return LLMResponse(
            text=txt,
            finish_reason=data["choices"][0].get("finish_reason"),
            total_tokens=self.tokenizer.count_tokens(txt),
        )

    def stream_chat(self, req: LLMRequest, **kwargs) -> Iterator[LLMStreamChunk]:
        """Performs a synchronous streaming chat, yielding LLMStreamChunk objects."""
        body = self._make_body(req)
        body["stream"] = True
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            with client.stream(
                "POST", "/chat/completions", json=body, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    chunk = _to_chunk(line)
                    if chunk is None:
                        continue
                    if chunk.is_finished:
                        yield chunk
                        return
                    # Correct token count using tokenizer (replaces rough byte-based estimate)
                    chunk.num_tokens = self.tokenizer.count_tokens(chunk.text)
                    yield chunk

    # ---------- Asynchronous methods ----------
    async def achat(self, req: LLMRequest, **kwargs) -> LLMResponse:
        """Performs an asynchronous non-streaming chat completion."""
        body = self._make_body(req)
        body["stream"] = False
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout
        ) as client:
            r = await client.post(
                "/chat/completions", json=body, headers=self._headers()
            )
            r.raise_for_status()
            data = r.json()
        txt: str = data["choices"][0]["message"]["content"]
        return LLMResponse(
            text=txt,
            finish_reason=data["choices"][0].get("finish_reason"),
            total_tokens=self.tokenizer.count_tokens(txt),
        )

    async def astream_chat(
        self, req: LLMRequest, **kwargs
    ) -> AsyncIterator[LLMStreamChunk]:
        """Performs an asynchronous streaming chat, yielding LLMStreamChunk objects."""
        body = self._make_body(req)
        body["stream"] = True
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout
        ) as client:
            async with client.stream(
                "POST", "/chat/completions", json=body, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    chunk = _to_chunk(line)
                    if chunk is None:
                        continue
                    if chunk.is_finished:
                        yield chunk
                        return
                    chunk.num_tokens = self.tokenizer.count_tokens(chunk.text)
                    yield chunk


# Example usage (for local testing)
if __name__ == "__main__":
    tok = HuggingFaceTokenizer("D:/Models/Qwen3-32B")
    import os

    conn = OpenAIConn(
        base_url="https://api.siliconflow.cn/v1",
        api_key=os.getenv("SILICON_API_KEY") or "",
        model="THUDM/GLM-Z1-9B-0414",
        tokenizer=tok,
    )

    # 1. Synchronous non-streaming test
    print("==================1. Test synchronous non-streaming==================")
    req = LLMRequest(messages=[{"role": "user", "content": "hello"}], max_tokens=64)
    print(conn.chat(req))

    # 2. Synchronous streaming test
    print("==================2. Test synchronous streaming==================")
    req = LLMRequest(num_tokens=100)
    for c in conn.stream_chat(req):
        # print(c.text, end="", flush=True)
        print(c.num_tokens, c.is_finished, c.finish_reason)
