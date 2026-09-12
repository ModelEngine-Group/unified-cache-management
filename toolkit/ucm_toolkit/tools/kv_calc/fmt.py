"""Output formatting (bordered tables + JSON)."""

import json as _json

from .detect import CLASS_LABELS

# ---------------------------------------------------------------------------
# Bordered-table primitives
# ---------------------------------------------------------------------------


def _bordered(rows, aligns=None, header=None):
    """Render a bordered table: '=' top/bottom rule, '| ... |' rows, no
    internal horizontal lines.

    rows: list of cell tuples (any arity). header: optional header tuple.
    aligns: per-column 'l'/'r'/'c' (default 'l').
    """
    all_rows = ([header] if header else []) + list(rows)
    if not all_rows:
        return ""
    ncol = len(all_rows[0])
    aligns = aligns or ["l"] * ncol
    widths = [0] * ncol
    for r in all_rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))

    def pad(s, w, a):
        s = str(s)
        return s.rjust(w) if a == "r" else (s.center(w) if a == "c" else s.ljust(w))

    def line(cells):
        return (
            "| "
            + " | ".join(pad(cells[i], widths[i], aligns[i]) for i in range(ncol))
            + " |"
        )

    body = [line(r) for r in all_rows]
    rule = "=" * len(body[0])
    return rule + "\n" + "\n".join(body) + "\n" + rule


def _grid_bordered(items, ncols=5):
    """Bordered row-major grid; trailing cells of the last row are blank."""
    if not items:
        return ""
    pad_n = (-len(items)) % ncols
    cells = list(items) + [""] * pad_n
    rows = [tuple(cells[i : i + ncols]) for i in range(0, len(cells), ncols)]
    return _bordered(rows, aligns=["l"] * ncols)


def _title(name):
    return f"\n{name}"


# --- size/bytes formatting ---
def _gib(b):
    return "n/a" if b is None else f"{b / (1024**3):.4f} GiB"


def _gib_seq(b):
    return "n/a" if b is None else f"{b / (1024**3):.4f} GiB/seq"


def _bytes(b):
    return "n/a" if b is None else f"{b:,.0f} B"


def _gb(b):
    return "n/a" if b is None else f"{b / 1e9:.4f} GB"


def format_bytes(b):
    return "n/a" if b is None else f"{b / (1024**3):.4f} GiB (= {b / 1e9:.4f} GB)"


# ---------------------------------------------------------------------------
# --list (bordered multi-column grid of model ids)
# ---------------------------------------------------------------------------


def render_preset_table(presets):
    ids = sorted(e["id"] for e in presets)
    return f"{len(ids)} preset models\n" + _grid_bordered(ids, ncols=5)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render_text(result):
    r = result
    model = r["model"]
    cls = r["classification"]
    params = r["params"]
    out = ["KV-CALC  —  KV cache size estimation"]

    tp, dp = params["tp"], params["dp"]
    total = r["total_bytes"]
    per_gpu_tp = total / tp if tp else 0  # per-GPU: divide by TP only (DP-free)
    input_len, num_req = params["tokens"], params["num_requests"]
    total_tokens = input_len * num_req
    seq_total = r["seq"].bytes_per_seq
    amort = seq_total / input_len if input_len else 0

    # ====================================================================
    # HEADLINE BLOCK — the only thing most users care about.
    #   tokens | size | per-GPU (÷ TP only; a request does not cross DP,
    #   so its KV cache is not scattered across DP ranks).
    # ====================================================================
    out.append("")
    out.append("KV CACHE")
    out.append(
        _bordered(
            [
                (f"{total_tokens:,}", f"tokens ({input_len:,} × {num_req:,})"),
                (_gib(total), f"size  ·  {_bytes(total)}  ({_gb(total)})"),
                (_gib(per_gpu_tp), f"per-GPU (÷TP={tp})"),
            ],
            aligns=["r", "l"],
            header=("Size", "Detail"),
        )
    )

    # ====================================================================
    # AUXILIARY BOX — everything else, in one bordered box.
    # Sub-headers marked with ◆ group the rows.
    # ====================================================================
    tag = {
        "registry": "[registry]",
        "prefix": "[prefix]",
        "inferred": "[INFERRED — verify]",
        "curated": "[curated preset]",
    }.get(cls.method, "")

    aux = []
    aux.append(("◆ Model", ""))
    aux.append((model.display_name, "name"))
    aux.append((model.source_desc, "source"))
    aux.append((model.loader_kind, "loader"))
    aux.append((cls.arch_string or "(none)", "architecture"))
    aux.append((f"{cls.label}  {tag}".rstrip(), "attention"))

    aux.append(("◆ Parameters", ""))
    aux.append((str(input_len), "input-len"))
    aux.append((str(num_req), "num-requests"))
    aux.append((f"{tp} / {dp}", "TP / DP"))
    aux.append((f"{params['kv_dtype']} ({params['kv_bytes']:.2f} B)", "kv-dtype"))
    aux.append(
        (
            f"{params['indexer_dtype']} ({params['indexer_bytes']:.2f} B)",
            "indexer-dtype",
        )
    )
    aux.append(("on" if params["gqa_copy"] else "off", "gqa-copy"))
    aux.append(
        ("on" if params["include_linear_state"] else "off", "include-linear-state")
    )

    aux.append(("◆ Other sizes", ""))
    aux.append((_gib(r["per_instance_bytes"]), f"per-instance (÷DP={dp})"))
    aux.append((_gib(r["per_gpu_bytes"]), f"per-GPU uniform (÷TP×DP={tp * dp})"))
    aux.append((_gib(r["per_seq_per_gpu"]), "per-request, per-GPU (÷TP)"))

    aux.append(("◆ Breakdown", ""))
    for p in r["seq"].parts:
        aux.append(
            (_gib_seq(p.bytes_per_seq), f"{p.name}  ·  {_bytes(p.bytes_per_seq)}")
        )
    aux.append((_gib_seq(seq_total), f"whole-model per-seq  ·  {_bytes(seq_total)}"))
    aux.append((f"{amort:,.2f} B/token", "amortized"))

    if r.get("v4_measured"):
        aux.append(("◆ DeepSeek V4 measured", ""))
        for m in r["v4_measured"]:
            dep = m["deployment"] + (" *" if m.get("selected") else "")
            aux.append(
                (
                    _gib_seq(m["per_seq_bytes"]),
                    f"{dep}  ·  {m['bytes_per_token']:,.4f} B/token × {input_len} "
                    f"= {_bytes(m['per_seq_bytes'])}  "
                    f"(per-GPU {_gib(m['per_seq_per_gpu'])}, total {_gib(m['total_bytes'])})",
                )
            )
        aux.append(("* = selected via --deployment", ""))

    if r.get("verbose_fields"):
        aux.append(("◆ Fields", ""))
        aux.append((str(r["verbose_fields"]), ""))

    out.append("")
    out.append("Details")
    out.append(_bordered(aux, aligns=["l", "l"]))

    # Notes (free-text warnings; kept below the box for readability).
    if r.get("notes"):
        out.append("")
        out.append("Notes")
        for n in r["notes"]:
            out.append(f"  • {n}")

    return "\n".join(out)


def render_json(result):
    r = result
    params = r["params"]
    model = r["model"]
    cls = r["classification"]
    payload = {
        "model": {
            "name": model.display_name,
            "source": model.source_desc,
            "loader": model.loader_kind,
            "architectures": model.architectures,
            "attention_class": cls.attention_class,
            "attention_label": cls.label,
            "classification_method": ("curated" if model.is_preset else cls.method),
        },
        "params": {
            "tokens": params["tokens"],
            "num_requests": params["num_requests"],
            "tp": params["tp"],
            "dp": params["dp"],
            "kv_dtype": params["kv_dtype"],
            "kv_bytes": params["kv_bytes"],
            "indexer_dtype": params["indexer_dtype"],
            "indexer_bytes": params["indexer_bytes"],
            "gqa_copy": params["gqa_copy"],
            "include_linear_state": params["include_linear_state"],
        },
        "cache": {
            "total_bytes": r["total_bytes"],
            "per_instance_bytes": r["per_instance_bytes"],
            "per_gpu_bytes": r["per_gpu_bytes"],
            "per_request_per_gpu_bytes": r["per_seq_per_gpu"],
            "units": {"gib": 1024**3, "gb": 1e9},
            "parts": [
                {
                    "name": p.name,
                    "bytes_per_seq": p.bytes_per_seq,
                    "shard": p.shard,
                    "heads_total": p.heads_total,
                }
                for p in r["seq"].parts
            ],
        },
        "v4_measured": [
            {
                "deployment": m["deployment"],
                "bytes_per_token": m["bytes_per_token"],
                "per_seq_bytes": m["per_seq_bytes"],
                "per_seq_per_gpu": m["per_seq_per_gpu"],
                "total_bytes": m["total_bytes"],
                "selected": m.get("selected", False),
            }
            for m in (r.get("v4_measured") or [])
        ],
        "notes": r.get("notes") or [],
    }
    return _json.dumps(payload, indent=2, ensure_ascii=False)
