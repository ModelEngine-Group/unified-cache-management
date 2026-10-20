"""Two real Chat turns for Decode writeback validation (standard library only).

This client measures TTFT; matching server key/load logs are the reuse proof.
It deliberately never sets ignore_eos=True or substitutes ground-truth history.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def assistant_history(message):
    """Preserve reasoning and tool fields instead of replaying content alone."""
    history = {"role": "assistant", "content": message.get("content") or ""}
    for name in ("reasoning", "reasoning_content", "tool_calls"):
        if message.get(name) is not None:
            history[name] = message[name]
    return history


def open_request(url, payload, timeout):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    req = urllib.request.Request(
        url,
        json.dumps(payload, ensure_ascii=False).encode(),
        headers=headers,
    )
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"HTTP {error.code}: {error.read().decode(errors='replace')}"
        ) from error


def stream_turn(url, payload, timeout):
    start = time.perf_counter()
    first_delta = first_content = None
    content = []
    usage = None
    request_id = None
    finish_reason = None
    with open_request(url, payload, timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"])
            request_id = event.get("id", request_id)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                now = time.perf_counter() - start
                if first_delta is None and any(
                    delta.get(key)
                    for key in (
                        "content",
                        "reasoning",
                        "reasoning_content",
                        "tool_calls",
                    )
                ):
                    first_delta = now
                if delta.get("content"):
                    if first_content is None:
                        first_content = now
                    content.append(delta["content"])
                finish_reason = choice.get("finish_reason") or finish_reason
    return dict(
        request_id=request_id,
        ttft_first_delta_s=first_delta,
        ttft_first_content_s=first_content,
        total_s=time.perf_counter() - start,
        usage=usage,
        content="".join(content),
        finish_reason=finish_reason,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Full /v1/chat/completions URL")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--min-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--second-max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0,
        help="Optional functional-test wait for asynchronous Store completion; not counted as TTFT",
    )
    parser.add_argument("--verify-id", default=None)
    parser.add_argument(
        "--chat-template-kwargs", default="{}", help="JSON, model/template specific"
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Please explain the history of computing in detailed chronological order. "
            "Include at least twenty milestones and discuss each in a full paragraph."
        ),
    )
    parser.add_argument(
        "--report", type=Path, help="Save request IDs, actual history and measurements"
    )
    args = parser.parse_args()
    if not 0 <= args.min_tokens < args.max_tokens:
        parser.error("Require 0 <= min-tokens < max-tokens")
    if not 0 <= args.settle_seconds <= 60:
        parser.error("Require 0 <= settle-seconds <= 60")
    verify_id = args.verify_id or str(uuid.uuid4())
    messages = [
        {
            "role": "system",
            "content": f"Validation session {verify_id}. Follow user instructions.",
        },
        {"role": "user", "content": args.prompt},
    ]
    common = dict(
        model=args.model,
        temperature=0,
        ignore_eos=False,
        chat_template_kwargs=json.loads(args.chat_template_kwargs),
    )
    start = time.perf_counter()
    with open_request(
        args.url,
        dict(
            common,
            messages=messages,
            stream=False,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        ),
        args.timeout,
    ) as response:
        first = json.load(response)
    first_seconds = time.perf_counter() - start
    choice = first["choices"][0]
    if choice["message"].get("tool_calls"):
        raise RuntimeError(
            "This text-only verifier does not execute tool calls; choose a text-only prompt"
        )
    history = assistant_history(choice["message"])
    if not any(
        history.get(key) for key in ("content", "reasoning", "reasoning_content")
    ):
        raise RuntimeError("Turn 1 returned no assistant text/reasoning to replay")
    messages2 = messages + [
        history,
        {"role": "user", "content": "Summarize the previous answer in one sentence."},
    ]
    if args.settle_seconds:
        time.sleep(args.settle_seconds)
    second = stream_turn(
        args.url,
        dict(
            common,
            messages=messages2,
            max_tokens=args.second_max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        ),
        args.timeout,
    )
    report = dict(
        verify_id=verify_id,
        model=args.model,
        turn1=dict(
            request_id=first.get("id"),
            usage=first.get("usage"),
            total_s=first_seconds,
            finish_reason=choice.get("finish_reason"),
        ),
        turn2=second,
        replayed_assistant=history,
        note="TTFT/usage alone do not prove reuse. Match Decode successful keys with next Prefill lookup and load logs.",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
