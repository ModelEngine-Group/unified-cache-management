#!/usr/bin/env bash
# Warm each request's 90% prompt prefix, then benchmark its full prompt.
# Run this script separately against HiCache and UCM linker servers.

set -Eeuo pipefail

if (( $# )); then
    echo 'This script takes no command-line arguments; edit its configuration block.' >&2
    exit 2
fi

# Edit this block before running. Keep these values consistent across runs.
BASE_URL=http://127.0.0.1:30000
MODEL=glm
TOKENIZER=/mnt/model/GLM-5.2-W4AFP8
LABEL=ucm                 # Change to hicache for the HiCache run.
OUTPUT_DIR=./bench-results
PYTHON=python3

# Workload parameters. Edit these values before each run.
NUM_PROMPTS=16
INPUT_LEN=16384
OUTPUT_LEN=100
SEED=42
MAX_CONCURRENCY=200
CACHE_HIT_RATIO=0.90
PAGE_SIZE=64
CACHE_SETTLE_SECONDS=10

[[ -n $MODEL ]] || { echo 'Set MODEL in the configuration block' >&2; exit 2; }
[[ -n $TOKENIZER ]] || { echo 'Set TOKENIZER in the configuration block' >&2; exit 2; }
for value in "$INPUT_LEN" "$OUTPUT_LEN" "$NUM_PROMPTS" "$PAGE_SIZE"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || {
        echo 'Input length, output length, request count, and page size must be positive integers' >&2
        exit 2
    }
done
[[ $SEED =~ ^[0-9]+$ ]] || { echo 'SEED must be a nonnegative integer' >&2; exit 2; }
[[ $CACHE_HIT_RATIO =~ ^0?\.[0-9]+$ ]] || {
    echo 'CACHE_HIT_RATIO must be between 0 and 1, such as 0.90' >&2
    exit 2
}
[[ $LABEL =~ ^[A-Za-z0-9_-]+$ ]] || {
    echo 'LABEL may contain only letters, digits, underscores and hyphens' >&2
    exit 2
}
command -v "$PYTHON" >/dev/null || { echo "Python not found: $PYTHON" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
RUN_DIR="$OUTPUT_DIR/${LABEL}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$RUN_DIR"

# Generate custom JSONL requests. Each warmup prompt is the first page-aligned
# CACHE_HIT_RATIO of the matching full prompt. SGLang's benchmark custom loader
# rejects fixed output lengths below 4, so the warmup requests use /generate
# directly to enforce exactly one generated token.
"$PYTHON" - "$TOKENIZER" "$RUN_DIR" "$NUM_PROMPTS" "$INPUT_LEN" \
    "$SEED" "$CACHE_HIT_RATIO" "$PAGE_SIZE" <<'PY'
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

tokenizer_path, run_dir, count, input_len, seed, ratio, page_size = sys.argv[1:]
run_dir = Path(run_dir)
count, input_len, seed, page_size = map(int, (count, input_len, seed, page_size))
ratio = float(ratio)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
rng = random.Random(seed)
words = (
    "amber birch cedar delta elm fern grove harbor indigo juniper kelp linen "
    "maple nectar olive pebble quartz river silver timber umber violet willow "
    "yellow zephyr archive beacon canvas domain engine forest garden horizon "
    "island journey kernel lantern meadow notice orbit planet query reason "
    "signal valley window"
).split()

def encode(text):
    return tokenizer.encode(text)

def decode(ids):
    return tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

def make_text(target, tag):
    """Build readable text whose tokenizer length is exactly target."""
    pieces = [tag]
    while len(encode(" ".join(pieces))) < target + 8:
        pieces.extend(rng.choices(words, k=2048))
    text = " ".join(pieces)
    for _ in range(32):
        ids = encode(text)
        if len(ids) >= target:
            text = decode(ids[:target])
        else:
            text += " " + " ".join(rng.choices(words, k=16))
        ids = encode(text)
        if len(ids) == target:
            return text, ids
    raise RuntimeError(
        f"Could not construct a {target}-token prompt for {tag}; "
        f"last encoded length was {len(encode(text))}."
    )

def common_prefix_len(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))

prefix_tokens = int(input_len * ratio) // page_size * page_size
if not 0 < prefix_tokens < input_len:
    raise ValueError("Target cache-hit ratio must leave both prefix and suffix tokens")

warmup_rows = []
test_rows = []
manifest = []
for index in range(count):
    test_text, test_ids = make_text(input_len, f"request_{index}_test")
    warm_text = decode(test_ids[:prefix_tokens])
    warm_ids = encode(warm_text)
    shared_tokens = common_prefix_len(warm_ids, test_ids)
    if len(warm_ids) < prefix_tokens - page_size or shared_tokens < prefix_tokens - page_size:
        raise RuntimeError(
            f"Request {index}: decoded warmup prompt retained only "
            f"{len(warm_ids)} tokens and {shared_tokens} shared tokens; "
            f"expected at least {prefix_tokens-page_size}."
        )

    warmup_rows.append({"prompt": warm_text})
    test_rows.append({
        "conversations": [
            {"from": "human", "value": test_text},
            {"from": "gpt", "value": "benchmark response"},
        ]
    })
    manifest.append({
        "request_index": index,
        "warmup_tokens": len(warm_ids),
        "test_tokens": len(test_ids),
        "shared_prefix_tokens": shared_tokens,
        "shared_prefix_ratio": shared_tokens / len(test_ids),
    })

with (run_dir / "warmup_prompts.jsonl").open("w", encoding="utf-8") as output:
    for row in warmup_rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
with (run_dir / "test_prompts.jsonl").open("w", encoding="utf-8") as output:
    for row in test_rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
with (run_dir / "prompt_manifest.jsonl").open("w", encoding="utf-8") as output:
    for row in manifest:
        output.write(json.dumps(row) + "\n")

actual_ratios = [row["shared_prefix_ratio"] for row in manifest]
print(
    f"Generated {count} prompt pairs; warmup input target={prefix_tokens}/{input_len} "
    f"tokens ({prefix_tokens/input_len:.2%}); actual shared-prefix ratio range="
    f"{min(actual_ratios):.2%}-{max(actual_ratios):.2%}."
)
PY

# Use the same native /generate route as bench_serving's SGLang backend. Unlike
# the custom dataset loader, this endpoint accepts max_new_tokens=1.
unset SGLANG_IS_IN_CI
echo "Results: $RUN_DIR"
echo "Pass 1/2: prefill each ${CACHE_HIT_RATIO} prompt prefix and generate 1 token"
set +e
"$PYTHON" - "$BASE_URL" "$RUN_DIR/warmup_prompts.jsonl" \
    >"$RUN_DIR/warmup.log" 2>&1 <<'PY'
import concurrent.futures
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

base_url, dataset_path = sys.argv[1:]
url = base_url.rstrip("/") + "/generate"
prompts = [
    json.loads(line)["prompt"]
    for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]

def warm_one(index, prompt):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1,
            "ignore_eos": True,
        },
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.loads(response.read().decode("utf-8"))
        print(f"Warmup {index + 1}/{len(prompts)} complete")
        return result
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Warmup request {index} failed: HTTP {error.code}: {detail}")

with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(prompts))) as pool:
    list(pool.map(lambda item: warm_one(*item), enumerate(prompts)))
PY
warmup_status=$?
set -e
if (( warmup_status != 0 )); then
    cat "$RUN_DIR/warmup.log" >&2
    exit 1
fi

echo "Waiting ${CACHE_SETTLE_SECONDS}s for asynchronous cache writes"
sleep "$CACHE_SETTLE_SECONDS"

COMMON=(
    --backend sglang
    --base-url "$BASE_URL"
    --model "$MODEL"
    --dataset-name custom
    --dataset-path "$RUN_DIR/test_prompts.jsonl"
    --tokenizer "$TOKENIZER"
    --num-prompts "$NUM_PROMPTS"
    --sharegpt-output-len "$OUTPUT_LEN"
    --seed "$SEED"
    --request-rate inf
    --max-concurrency "$MAX_CONCURRENCY"
    --warmup-requests 0
    --output-details
)

echo "Pass 2/2: measure the full ${INPUT_LEN}-token prompts"
"$PYTHON" -m sglang.benchmark.serving "${COMMON[@]}" \
    --cache-report \
    --output-file "$RUN_DIR/hit.jsonl" \
    >"$RUN_DIR/hit.log" 2>&1 || {
        cat "$RUN_DIR/hit.log" >&2
        exit 1
    }

"$PYTHON" - "$RUN_DIR/hit.jsonl" "$RUN_DIR/requests.csv" "$NUM_PROMPTS" "$OUTPUT_LEN" <<'PY'
import csv
import json
import statistics
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    records = [line for line in source if line.strip()]
result = json.loads(records[-1])
ttfts = result.get("ttfts", [])
itls = result.get("itls", [])
inputs = result.get("input_lens", [])
outputs = result.get("output_lens", [])
details = result.get("cached_tokens_details", [])
cached = result.get("cached_tokens", [])
errors = result.get("errors", [])
expected_requests = int(sys.argv[3])
expected_output_len = int(sys.argv[4])

if len(ttfts) != expected_requests or any(errors):
    raise SystemExit(
        f"Expected {expected_requests} successful requests; got {len(ttfts)} records, "
        f"errors={errors}. Inspect the benchmark log and raw JSONL."
    )

def ms_mean(values):
    return round(statistics.mean(values) * 1000, 3) if values else ""

rows = []
for index, ttft in enumerate(ttfts):
    gaps = itls[index] if index < len(itls) else []
    hit = details[index] if index < len(details) else None
    hit = hit or {}
    rows.append({
        "request_index": index,
        "input_tokens": inputs[index] if index < len(inputs) else "",
        "output_tokens": outputs[index] if index < len(outputs) else "",
        "cached_tokens": cached[index] if index < len(cached) else "",
        "ttft_ms": round(ttft * 1000, 3),
        "mean_itl_ms": ms_mean(gaps),
        "first_10_itl_ms": ms_mean(gaps[:10]),
        "last_10_itl_ms": ms_mean(gaps[-10:]),
        "device_hit_tokens": hit.get("device", ""),
        "host_hit_tokens": hit.get("host", ""),
        "storage_hit_tokens": hit.get("storage", ""),
    })

with open(sys.argv[2], "w", newline="", encoding="utf-8") as target:
    writer = csv.DictWriter(target, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Completed requests: {len(rows)}")
print(f"Mean TTFT: {result.get('mean_ttft_ms', 'n/a')} ms")
print(f"Mean TPOT: {result.get('mean_tpot_ms', 'n/a')} ms")
print(f"Per-request CSV: {sys.argv[2]}")
if any(length != expected_output_len for length in outputs):
    print(f"WARNING: observed output lengths differ from {expected_output_len}: {outputs}")
if not any(details):
    print("WARNING: no per-request cache breakdown was returned by the server")
total_input = sum(inputs)
total_cached = sum(cached)
observed_ratio = total_cached / total_input if total_input else 0.0
print(f"Observed cached-token ratio: {observed_ratio:.2%}")
if observed_ratio < 0.85:
    print(
        "WARNING: observed overall cache-hit ratio is below 85%; inspect cache "
        "availability, prompt_manifest.jsonl, and per-request hit counts."
    )
PY

echo "Raw benchmark output: $RUN_DIR/hit.jsonl"
echo "Benchmark log: $RUN_DIR/hit.log"
echo "Warmup prompts and expected prefix ratios: $RUN_DIR/prompt_manifest.jsonl"
