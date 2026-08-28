Run the same long-prefix request twice. The first request populates UCM; the
second request is eligible for an external-cache hit.

```bash
python3 - <<'PY'
import json
import os
import urllib.request

model = os.environ["MODEL_ALIAS"]
prompt = (
    "Unified Cache Management reuses a stable prefix across requests. " * 160
    + "Summarize the preceding context in one sentence."
)
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 32,
    "temperature": 0,
}).encode()

for attempt in (1, 2):
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    print(f"request {attempt}: {result['choices'][0]['message']['content']}")
PY
```

A successful check has two parts:

1. Both requests return HTTP 200 with a non-empty `choices` array.
2. The serving logs show a UCM lookup/load on the second request. Exact log
   wording varies by UCM and engine version; confirm that the matched or loaded
   token count is greater than zero.

If the second request misses, confirm that both requests used identical model,
tokenizer, chat template, block size, and prefix text, and that `/mnt/ucm` is
writable by the serving process.

