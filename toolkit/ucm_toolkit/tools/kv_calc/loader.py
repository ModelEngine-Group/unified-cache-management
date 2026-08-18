"""Model config loading (pure stdlib: ``urllib`` + ``json``).

Sources, in autodetect order: preset alias -> local path -> ``hf://`` or bare
``org/model`` -> HuggingFace hub; ``ms://`` -> ModelScope. Remote fetch is a
convenience; presets (offline) and ``--model-dir`` (local config.json) are the
reliable paths and need no network.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request

from .presets import get_preset

_HF_RESOLVE = "https://huggingface.co/{mid}/resolve/main/config.json"
_MS_RAW = "https://www.modelscope.cn/models/{mid}/resolve/master/config.json"
_MS_API = (
    "https://www.modelscope.cn/api/v1/models/{mid}/repo"
    "?Revision=master&FilePath=config.json"
)
_USER_AGENT = "ucm-toolkit/kv-calc"
_TIMEOUT = 30


class LoadError(Exception):
    """Raised when a model config cannot be resolved or parsed."""


class LoadedModel:
    """A flattened, field-ready model config."""

    __slots__ = (
        "fields",
        "architectures",
        "model_type",
        "source_desc",
        "loader_kind",
        "is_preset",
        "preset_entry",
    )

    def __init__(
        self,
        fields,
        architectures,
        model_type,
        source_desc,
        loader_kind,
        is_preset=False,
        preset_entry=None,
    ):
        self.fields = fields
        self.architectures = architectures
        self.model_type = model_type
        self.source_desc = source_desc
        self.loader_kind = loader_kind  # "preset" | "json" | "hf" | "ms"
        self.is_preset = is_preset
        self.preset_entry = preset_entry

    @property
    def display_name(self):
        if self.preset_entry:
            return self.preset_entry["id"]
        return self.source_desc


def load_config(model_spec, source=None, model_dir=None):
    """Resolve and load a model config.

    ``source`` forces one of ``preset``/``local``/``hf``/``ms``. ``model_dir``
    is a shortcut for loading a local config.json directory. Raises
    :class:`LoadError` on failure.
    """
    if model_dir:
        return _from_local(model_dir)
    if source == "preset":
        return _from_preset_name(model_spec)
    if source == "local":
        return _from_local(model_spec)
    if source == "hf":
        return _from_hf(_strip_scheme(model_spec, "hf"))
    if source == "ms":
        return _from_ms(_strip_scheme(model_spec, "ms"))

    # Autodetect.
    entry = get_preset(model_spec)
    if entry is not None:
        return _from_preset(entry)
    spec = model_spec.strip()
    if spec.startswith("ms://"):
        return _from_ms(spec[len("ms://") :])
    if spec.startswith("hf://"):
        return _from_hf(spec[len("hf://") :])
    if os.path.exists(spec):
        return _from_local(spec)
    if "/" in spec and " " not in spec:
        return _from_hf(spec)
    raise LoadError(
        f"cannot resolve --model {model_spec!r}: not a preset, not a path, "
        f"and not an obvious HF id. Pass --source preset|local|hf|ms."
    )


def _from_preset_name(name):
    entry = get_preset(name)
    if entry is None:
        raise LoadError(f"no preset named {name!r}; use --list to see options")
    return _from_preset(entry)


def _from_preset(entry):
    return LoadedModel(
        fields=entry["fields"],
        architectures=list(entry.get("architectures") or []),
        model_type=None,
        source_desc=f"preset:{entry['id']}",
        loader_kind="preset",
        is_preset=True,
        preset_entry=entry,
    )


def _from_local(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        config_path = os.path.join(path, "config.json")
    else:
        config_path = path
    if not os.path.exists(config_path):
        raise LoadError(f"no config.json at {config_path!r}")
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    fields, architectures, model_type = _flatten(raw)
    return LoadedModel(
        fields=fields,
        architectures=architectures,
        model_type=model_type,
        source_desc=f"local:{path}",
        loader_kind="json",
    )


def _from_hf(model_id):
    model_id = model_id.strip().rstrip("/")
    try:
        raw = _fetch_json(_HF_RESOLVE.format(mid=model_id))
    except Exception as exc:  # network / HTTP / parse
        raise LoadError(f"failed to fetch HF config for {model_id!r}: {exc}") from exc
    fields, architectures, model_type = _flatten(raw)
    return LoadedModel(
        fields=fields,
        architectures=architectures,
        model_type=model_type,
        source_desc=f"hf:{model_id}",
        loader_kind="hf",
    )


def _from_ms(model_id):
    model_id = model_id.strip().rstrip("/")
    raw = None
    err = None
    try:
        raw = _fetch_json(_MS_RAW.format(mid=model_id))
    except Exception as exc:  # noqa: BLE001 - fall back to API endpoint
        err = exc
    if raw is None:
        try:
            payload = _fetch_json(_MS_API.format(mid=model_id))
            content = (payload.get("Data") or {}).get("Content")
            if not content:
                raise LoadError(f"ModelScope API returned no config for {model_id!r}")
            raw = json.loads(base64.b64decode(content))
        except LoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LoadError(
                f"failed to fetch ModelScope config for {model_id!r}: "
                f"raw={err}, api={exc}"
            ) from exc
    fields, architectures, model_type = _flatten(raw)
    return LoadedModel(
        fields=fields,
        architectures=architectures,
        model_type=model_type,
        source_desc=f"ms:{model_id}",
        loader_kind="ms",
    )


def _fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _flatten(raw):
    """Flatten ``text_config`` into the top-level dict (text_config wins for
    attention fields; architectures/model_type taken from top first)."""
    if not isinstance(raw, dict):
        raise LoadError("config is not a JSON object")
    top = dict(raw)
    text = top.get("text_config")
    if isinstance(text, dict):
        flat = dict(top)
        flat.pop("text_config", None)
        flat.update(
            {k: v for k, v in text.items() if k not in ("architectures", "model_type")}
        )
        architectures = top.get("architectures") or text.get("architectures") or []
        model_type = top.get("model_type") or text.get("model_type")
        flat["architectures"] = architectures
        flat["model_type"] = model_type
        return flat, list(architectures or []), model_type
    return top, list(top.get("architectures") or []), top.get("model_type")


def _strip_scheme(spec, scheme):
    s = spec.strip()
    prefix = scheme + "://"
    return s[len(prefix) :] if s.startswith(prefix) else s
