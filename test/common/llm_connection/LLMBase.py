from __future__ import annotations

from dataclasses import dataclass
from typing import (
    AsyncIterator,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)


@dataclass
class LLMRequest:
    """
    Either `messages` or `num_tokens` should be provided (not both).

    - `messages`: A standard list of message dictionaries (e.g., with roles like 'user', 'assistant').
    - `num_tokens`: If provided, the system will generate random messages totaling approximately this many tokens.
    - `max_tokens`: Maximum number of output tokens to generate.
    - `ignore_eos`: If True, generation will continue past the end-of-sequence token (only respected by vLLM).
    - `temperature`: Sampling temperature (default: 0.0 for deterministic output).
    - `top_p`: Nucleus sampling parameter (default: 1.0 for full distribution).
    - `timeout`: Request timeout in seconds (default: 600.0).
    """

    messages: Sequence[dict] = ()
    num_tokens: Optional[int] = None
    ignore_eos: bool = False
    max_tokens: Optional[int] = None
    temperature: float = 0.0
    top_p: float = 1.0
    timeout: float = 600.0


@dataclass
class LLMResponse:
    """Represents a complete response from an LLM."""

    text: str
    finish_reason: Optional[str]
    total_tokens: int


@dataclass
class LLMStreamChunk:
    """Represents a single streaming chunk during LLM generation."""

    text: str
    num_tokens: int
    is_finished: bool
    finish_reason: Optional[str]


@runtime_checkable
class LLMConnection(Protocol):
    """
    Minimal contract for LLM clients.

    Any connector that implements these four methods satisfies the LLMConnection protocol,
    without needing to inherit from a base class.

    - `chat` and `achat`: Perform single-turn synchronous and asynchronous inference, respectively.
    - `stream_chat` and `astream_chat`: Yield structured `LLMStreamChunk` objects during generation.
    """

    def chat(self, request: LLMRequest, **kwargs) -> LLMResponse: ...

    """Synchronous single-turn chat completion."""

    def stream_chat(
        self, request: LLMRequest, **kwargs
    ) -> Iterator[LLMStreamChunk]: ...

    """Synchronous streaming chat that yields structured generation chunks."""

    async def achat(self, request: LLMRequest, **kwargs) -> LLMResponse: ...

    """Asynchronous single-turn chat completion."""

    async def astream_chat(
        self, request: LLMRequest, **kwargs
    ) -> AsyncIterator[LLMStreamChunk]: ...

    """Asynchronous streaming chat that yields structured generation chunks."""

    # TODO: Consider adding a unified calling interface.
