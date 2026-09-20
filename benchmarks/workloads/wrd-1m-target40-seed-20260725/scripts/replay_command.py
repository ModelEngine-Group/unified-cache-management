#!/usr/bin/env python3
"""Print a future replay command. This script never executes it or sends HTTP."""
import argparse
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.inputs.read_text())
    for key in ["runner", "order", "manifest", "profile", "parquet", "tools", "tokenizer"]:
        path = Path(config[key])
        config[key] = str(path if path.is_absolute() else ROOT / path)
    if not args.results_dir.is_absolute():
        parser.error("--results-dir must be absolute")
    command = [args.python, "-B", config["runner"]]
    for name in ["order", "manifest", "profile", "parquet", "tools", "tokenizer", "max_concurrency",
                 "session_pool", "seed", "prompt_salt_prefix", "max_model_len", "max_output_tokens", "request_timeout"]:
        command += ["--" + name.replace("_", "-"), str(config[name])]
    for name, value in [("base-url", args.base_url), ("model", args.model), ("run-id", args.run_id),
                        ("output", args.results_dir / "result.json"), ("trace-output", args.results_dir / "trace.jsonl")]:
        command += ["--" + name, str(value)]
    print(shlex.join(command))


if __name__ == "__main__":
    main()
