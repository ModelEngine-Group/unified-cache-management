import ast
import glob
import gzip
import json
import os
import re
import sys

match_pattern = re.compile(
    r"timestamp:\s*(?P<timestamp>\d+\.\d+),\s*"
    r"input_length:\s*(?P<input_length>\d+),\s*"
    r"output_length:\s*(?P<output_length>\d+),\s*"
    r"ucm_block_ids:\s*(?P<ucm_block_ids>\[.*?\])"
)

output_log = "./output.jsonl"


def process_file(file_path):
    count = 0
    is_gz = file_path.endswith(".log.gz")

    if is_gz:
        opener = lambda: gzip.open(file_path, "rt", encoding="utf-8", errors="ignore")
    else:
        opener = lambda: open(file_path, "r", encoding="utf-8")

    with opener() as f:
        for line in f:
            match = match_pattern.search(line)
            if match:
                clean_hash_ids = ast.literal_eval(match.group("ucm_block_ids"))
                data = {
                    "timestamp": float(match.group("timestamp")),
                    "input_length": int(match.group("input_length")),
                    "output_length": int(match.group("output_length")),
                    "hash_ids": clean_hash_ids,
                }
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1
    return count


if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    print("Usage: python gene_trace.py <directory_or_log_file>")
    sys.exit(1)

if os.path.exists(output_log):
    os.remove(output_log)

if os.path.isfile(target):
    files = [target]
elif os.path.isdir(target):
    dir_pattern = os.path.join(target, "ucm.log*")
    gz_pattern = os.path.join(target, "ucm_*.log.gz")
    files = glob.glob(dir_pattern) + sorted(glob.glob(gz_pattern))
    if not files:
        print(f"No log files found in: {target}")
        sys.exit(1)
else:
    print(f"Log files not found: {target}")
    sys.exit(1)

print(f"Processing log {len(files)} file(s)... ")

with open(output_log, "a") as out_f:
    total = 0
    for file_path in files:
        print(f"  -> {os.path.basename(file_path)}")
        count = process_file(file_path)
        total += count
        print(f"{count} records extracted from {file_path}")

print(f"Done! Total: {total} records -> {output_log}")
