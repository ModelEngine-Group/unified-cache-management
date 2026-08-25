#!/usr/bin/env python3
"""Resolve a role-local vLLM KV transfer template at Pod startup.

The Helm chart writes two JSON files for every role:

``kv-transfer.template.json``
    A complete vLLM ``--kv-transfer-config`` object. Dynamic identity values
    are represented by the exact string sentinels ``__UC_ENGINE_ID__`` and
    ``__UC_KV_PORT__``.

``kv-transfer.meta.json``
    Resolver metadata with this contract (schema version 1)::

        {
          "schemaVersion": 1,
          "groupNamePrefix": "<ModelServing name>-",
          "roleName": "prefill",
          "dynamicIdentity": true,
          "connector": "MooncakeConnectorV1",
          "roleKind": "producer",
          "prefillReplicas": 2,
          "decodeReplicas": 2,
          "portSpan": 8,
          "identity": {
            "engineIdBase": 0,
            "kvPortBase": 36000,
            "instanceStride": 100
          }
        }

For NIXL, ``identity`` contains only ``engineIdBase`` and the template and
metadata contain no KV-port fields. For a static non-PD ``UCMConnector``
template, ``dynamicIdentity`` is false and ``identity`` is omitted; Pod
identity labels are then not required.

The group and role labels are injected through the Downward API as
``UC_PD_GROUP_NAME`` and ``UC_PD_ROLE_ID``. ``WORKER_INDEX`` is intentionally
never read: an entry process and all workers belonging to the same role
replica must resolve to the same engine ID and port base.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ENGINE_ID_SENTINEL = "__UC_ENGINE_ID__"
KV_PORT_SENTINEL = "__UC_KV_PORT__"
_SENTINEL_PATTERN = re.compile(r"__UC_[A-Z0-9_]+__")
_V1_ONLY_FIELDS = ("kv_rank", "kv_buffer_device", "kv_parallel_size")


class ResolutionError(ValueError):
    """Raised when the template, metadata, or Pod identity is invalid."""


def _is_int(value: Any) -> bool:
    """Return true for JSON integers while rejecting booleans."""

    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(
    owner: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if key not in owner:
        raise ResolutionError(f"missing required integer field: {key}")
    value = owner[key]
    if not _is_int(value):
        raise ResolutionError(f"{key} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ResolutionError(f"{key} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ResolutionError(f"{key} must be <= {maximum}, got {value}")
    return value


def _require_nonempty_string(owner: Mapping[str, Any], key: str) -> str:
    value = owner.get(key)
    if not isinstance(value, str) or not value:
        raise ResolutionError(f"{key} must be a non-empty string")
    return value


def _parse_ordinal(value: str | None, prefix: str, label_name: str) -> int:
    if value is None or value == "":
        raise ResolutionError(f"missing Pod label {label_name}")
    pattern = re.compile(rf"^{re.escape(prefix)}(0|[1-9][0-9]*)$")
    match = pattern.fullmatch(value)
    if match is None:
        raise ResolutionError(
            f"malformed Pod label {label_name}={value!r}; "
            f"expected {prefix!r} followed by a non-negative ordinal"
        )
    return int(match.group(1))


def _walk_strings(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield f"{path}.<key:{key}>", key
            yield from _walk_strings(item, f"{path}.{key}")


def _count_exact_string(value: Any, expected: str) -> int:
    return sum(1 for _, candidate in _walk_strings(value) if candidate == expected)


def _replace_exact_values(value: Any, replacements: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_exact_values(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_exact_values(item, replacements)
            for key, item in value.items()
        }
    return value


def _find_key_paths(value: Any, wanted: str, path: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_find_key_paths(item, wanted, f"{path}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key == wanted:
                paths.append(child_path)
            paths.extend(_find_key_paths(item, wanted, child_path))
    return paths


def _connector_children(config: Mapping[str, Any], path: str) -> list[Mapping[str, Any]]:
    extra = config.get("kv_connector_extra_config")
    if extra is None:
        return []
    if not isinstance(extra, dict):
        raise ResolutionError(f"{path}.kv_connector_extra_config must be an object")
    if "connectors" not in extra:
        return []
    children = extra["connectors"]
    if not isinstance(children, list) or not children:
        raise ResolutionError(
            f"{path}.kv_connector_extra_config.connectors must be a non-empty list"
        )
    for index, child in enumerate(children):
        if not isinstance(child, dict):
            raise ResolutionError(f"{path}.connectors[{index}] must be an object")
    return children


def _iter_connector_configs(
    config: Mapping[str, Any], path: str = "root"
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    connector = config.get("kv_connector")
    if not isinstance(connector, str) or not connector:
        raise ResolutionError(f"{path}.kv_connector must be a non-empty string")
    yield path, config
    for index, child in enumerate(_connector_children(config, path)):
        yield from _iter_connector_configs(child, f"{path}.connectors[{index}]")


def _validate_transport_semantics(
    config: Mapping[str, Any],
    *,
    connector: str,
    role_kind: str,
    path: str,
) -> None:
    expected_role = f"kv_{role_kind}"
    if config.get("kv_role") != expected_role:
        raise ResolutionError(
            f"{path}.kv_role must be {expected_role!r} for metadata "
            f"roleKind={role_kind!r}"
        )

    if connector == "MooncakeConnectorV1":
        expected_rank = 0 if role_kind == "producer" else 1
        rank = config.get("kv_rank")
        if not _is_int(rank) or rank != expected_rank:
            raise ResolutionError(
                f"{path}.kv_rank must be {expected_rank} for "
                f"MooncakeConnectorV1 {role_kind}"
            )
    elif connector == "MooncakeHybridConnector":
        residuals = [field for field in _V1_ONLY_FIELDS if field in config]
        if residuals:
            raise ResolutionError(
                f"{path} MooncakeHybridConnector must not contain V1-only "
                f"field(s): {', '.join(residuals)}"
            )
    elif connector == "NixlConnector":
        residuals = [field for field in _V1_ONLY_FIELDS if field in config]
        if residuals:
            raise ResolutionError(
                f"{path} NixlConnector must not contain Mooncake V1-only "
                f"field(s): {', '.join(residuals)}"
            )


def _validate_dynamic_shape(
    template: Mapping[str, Any], has_port: bool, role_kind: str
) -> None:
    connector = template.get("kv_connector")
    engine_paths = _find_key_paths(template, "engine_id")
    port_paths = _find_key_paths(template, "kv_port")

    if connector == "NixlConnector":
        if has_port:
            raise ResolutionError("NixlConnector identity must not contain port fields")
        if engine_paths != ["$.engine_id"] or port_paths:
            raise ResolutionError(
                "NixlConnector must contain only the root engine_id identity field"
            )
        _validate_transport_semantics(
            template,
            connector=connector,
            role_kind=role_kind,
            path="root",
        )
        return

    if connector in {"MooncakeConnectorV1", "MooncakeHybridConnector"}:
        if not has_port:
            raise ResolutionError(f"{connector} identity requires KV-port fields")
        if engine_paths != ["$.engine_id"] or port_paths != ["$.kv_port"]:
            raise ResolutionError(
                f"{connector} must contain root engine_id and root kv_port only"
            )
        _validate_transport_semantics(
            template,
            connector=connector,
            role_kind=role_kind,
            path="root",
        )
        return

    if connector == "MultiConnector":
        if not has_port:
            raise ResolutionError("MultiConnector identity requires KV-port fields")
        children = _connector_children(template, "root")
        if role_kind != "producer" or template.get("kv_role") != "kv_producer":
            raise ResolutionError(
                "MultiConnector is valid only for a producer and must have "
                "root.kv_role='kv_producer'"
            )
        root_residuals = [field for field in _V1_ONLY_FIELDS if field in template]
        if root_residuals:
            raise ResolutionError(
                "MultiConnector root must not contain transport-only field(s): "
                + ", ".join(root_residuals)
            )
        if engine_paths != ["$.engine_id"]:
            raise ResolutionError(
                "MultiConnector must contain exactly one engine_id at the root"
            )
        if "kv_port" in template:
            raise ResolutionError("MultiConnector root must not contain kv_port")

        mooncake_children = [
            (index, child)
            for index, child in enumerate(children)
            if child.get("kv_connector")
            in {"MooncakeConnectorV1", "MooncakeHybridConnector"}
        ]
        ucm_children = [
            child for child in children if child.get("kv_connector") == "UCMConnector"
        ]
        if len(mooncake_children) != 1 or len(ucm_children) != 1 or len(children) != 2:
            raise ResolutionError(
                "MultiConnector must contain exactly one Mooncake child and one "
                "UCMConnector child"
            )
        mooncake_index, mooncake = mooncake_children[0]
        expected_port_path = (
            f"$.kv_connector_extra_config.connectors[{mooncake_index}].kv_port"
        )
        if port_paths != [expected_port_path]:
            raise ResolutionError(
                "MultiConnector must contain exactly one kv_port on its Mooncake child"
            )
        if "engine_id" in mooncake or any(
            "engine_id" in child or "kv_port" in child for child in ucm_children
        ):
            raise ResolutionError(
                "MultiConnector children must not duplicate identity fields"
            )
        mooncake_connector = mooncake.get("kv_connector")
        assert isinstance(mooncake_connector, str)
        _validate_transport_semantics(
            mooncake,
            connector=mooncake_connector,
            role_kind="producer",
            path=f"root.connectors[{mooncake_index}]",
        )
        ucm_index = next(
            index
            for index, child in enumerate(children)
            if child.get("kv_connector") == "UCMConnector"
        )
        if children[ucm_index].get("kv_role") != "kv_both":
            raise ResolutionError(
                f"root.connectors[{ucm_index}].kv_role must be 'kv_both' "
                "for UCMConnector"
            )
        return

    raise ResolutionError(
        f"dynamic identity is not supported for connector {connector!r}"
    )


def _selected_connector_name(template: Mapping[str, Any]) -> str:
    """Return the user-selected transport connector represented by a template.

    A producer with UCM has ``MultiConnector`` at the root, while metadata keeps
    the original vLLM transport connector name. This preserves the public
    distinction between the selected connector and the chart-generated wrapper.
    """

    root = template.get("kv_connector")
    if root != "MultiConnector":
        assert isinstance(root, str)  # validated by _iter_connector_configs
        return root
    children = _connector_children(template, "root")
    selected = [
        child.get("kv_connector")
        for child in children
        if child.get("kv_connector") != "UCMConnector"
    ]
    if len(selected) != 1 or not isinstance(selected[0], str):
        raise ResolutionError(
            "MultiConnector metadata requires exactly one non-UCM child connector"
        )
    return selected[0]


def _validate_sentinels(
    template: Mapping[str, Any],
    *,
    identity_enabled: bool,
    has_port: bool,
) -> None:
    expected_engine = 1 if identity_enabled else 0
    expected_port = 1 if has_port else 0
    actual_engine = _count_exact_string(template, ENGINE_ID_SENTINEL)
    actual_port = _count_exact_string(template, KV_PORT_SENTINEL)
    if actual_engine != expected_engine or actual_port != expected_port:
        raise ResolutionError(
            "template sentinel count mismatch: "
            f"engineId expected {expected_engine}, found {actual_engine}; "
            f"kvPort expected {expected_port}, found {actual_port}"
        )


def _ensure_no_residual_sentinels(config: Mapping[str, Any]) -> None:
    residuals: list[str] = []
    for path, value in _walk_strings(config):
        if _SENTINEL_PATTERN.search(value):
            residuals.append(f"{path}={value!r}")
    if residuals:
        raise ResolutionError(
            "unresolved KV-transfer sentinel(s): " + ", ".join(residuals)
        )


@contextmanager
def _redirect_process_stdout_to_stderr() -> Iterable[None]:
    """Keep both Python and native plugin diagnostics off stdout.

    ``redirect_stdout`` alone cannot catch C extensions or libraries writing
    directly to file descriptor 1. Temporarily mapping fd 1 to fd 2 preserves
    stdout as the resolver's single-JSON-line API.
    """

    try:
        sys.stdout.flush()
    except (AttributeError, OSError):
        pass
    saved_stdout_fd = os.dup(1)
    try:
        os.dup2(2, 1)
        with redirect_stdout(sys.stderr):
            yield
    finally:
        # Flush libc stdio while fd 1 still targets stderr. Without this, a
        # plugin's buffered printf() can be emitted only after fd 1 is restored.
        try:
            import ctypes

            libc = ctypes.CDLL(None)
            libc.fflush.argtypes = [ctypes.c_void_p]
            libc.fflush.restype = ctypes.c_int
            libc.fflush(None)
        except Exception:
            # The chart runtime uses --output, so even an exotic C runtime with
            # an inaccessible flush API cannot corrupt the resolved JSON file.
            pass
        try:
            sys.stdout.flush()
        except (AttributeError, OSError):
            pass
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)


def _probe_connector_registry(config: Mapping[str, Any]) -> None:
    """Load vLLM plugins, then resolve every connector through its factory.

    We deliberately do not translate or fall back between connector names. In
    particular, a missing ``MooncakeHybridConnector`` remains a startup error.
    Python- and fd-level stdout redirection preserves the resolver's stdout as
    a JSON-only API even if a third-party native plugin prints while imported.
    """

    try:
        with _redirect_process_stdout_to_stderr():
            from vllm.plugins import load_general_plugins

            load_general_plugins()
            from vllm.distributed.kv_transfer.kv_connector.factory import (
                KVConnectorFactory,
            )

            for path, connector_config in _iter_connector_configs(config):
                name = connector_config["kv_connector"]
                module_path = connector_config.get("kv_connector_module_path")
                probe_config = SimpleNamespace(
                    kv_connector=name,
                    kv_connector_module_path=module_path,
                )
                try:
                    KVConnectorFactory.get_connector_class(probe_config)
                except Exception as exc:
                    raise ResolutionError(
                        f"connector registry validation failed at {path} for "
                        f"{name!r}: {type(exc).__name__}: {exc}"
                    ) from exc
    except ResolutionError:
        raise
    except Exception as exc:
        raise ResolutionError(
            "unable to load vLLM plugins/KVConnectorFactory: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def resolve_kv_transfer_config(
    template: Mapping[str, Any],
    meta: Mapping[str, Any],
    *,
    group_name: str | None,
    role_id: str | None,
    probe_registry: bool = True,
) -> dict[str, Any]:
    """Resolve and validate one vLLM KV transfer configuration."""

    if not isinstance(template, dict):
        raise ResolutionError("KV-transfer template must be a JSON object")
    if not isinstance(meta, dict):
        raise ResolutionError("KV-transfer metadata must be a JSON object")
    schema_version = _require_int(meta, "schemaVersion", minimum=1)
    if schema_version != SCHEMA_VERSION:
        raise ResolutionError(
            f"unsupported metadata schemaVersion {schema_version}; "
            f"expected {SCHEMA_VERSION}"
        )

    # Validate connector tree even when the registry probe is disabled. This
    # keeps malformed generated JSON from reaching vLLM in unit/offline tests.
    list(_iter_connector_configs(template))

    dynamic_identity = meta.get("dynamicIdentity")
    if not isinstance(dynamic_identity, bool):
        raise ResolutionError("dynamicIdentity must be a boolean")
    meta_connector = _require_nonempty_string(meta, "connector")
    selected_connector = _selected_connector_name(template)
    if meta_connector != selected_connector:
        raise ResolutionError(
            f"metadata connector {meta_connector!r} does not match template "
            f"transport connector {selected_connector!r}"
        )

    identity = meta.get("identity")
    if not dynamic_identity:
        if identity not in (None, {}):
            raise ResolutionError(
                "identity must be omitted, null, or empty when dynamicIdentity=false"
            )
        _validate_sentinels(
            template,
            identity_enabled=False,
            has_port=False,
        )
        resolved = deepcopy(template)
    else:
        if not isinstance(identity, dict):
            raise ResolutionError("identity must be an object or null")
        expected_identity_fields = {"engineIdBase"}
        if "kvPortBase" in identity or "instanceStride" in identity:
            expected_identity_fields.update({"kvPortBase", "instanceStride"})
        unexpected_identity_fields = sorted(set(identity) - expected_identity_fields)
        if unexpected_identity_fields:
            raise ResolutionError(
                "identity contains unsupported field(s): "
                + ", ".join(unexpected_identity_fields)
            )
        engine_id_base = _require_int(identity, "engineIdBase", minimum=0)
        prefill_replicas = _require_int(meta, "prefillReplicas", minimum=1)
        decode_replicas = _require_int(meta, "decodeReplicas", minimum=1)
        group_prefix = _require_nonempty_string(meta, "groupNamePrefix")
        role_name = _require_nonempty_string(meta, "roleName")
        role_kind = meta.get("roleKind")
        if role_kind not in {"producer", "consumer"}:
            raise ResolutionError("roleKind must be exactly 'producer' or 'consumer'")

        group_ordinal = _parse_ordinal(
            group_name, group_prefix, "modelserving.volcano.sh/group-name"
        )
        role_ordinal = _parse_ordinal(
            role_id, f"{role_name}-", "modelserving.volcano.sh/role-id"
        )
        role_replicas = (
            prefill_replicas if role_kind == "producer" else decode_replicas
        )
        if role_ordinal >= role_replicas:
            raise ResolutionError(
                f"role ordinal {role_ordinal} is outside {role_kind} replica "
                f"range 0..{role_replicas - 1}"
            )

        per_group = prefill_replicas + decode_replicas
        role_offset = (
            role_ordinal
            if role_kind == "producer"
            else prefill_replicas + role_ordinal
        )
        instance_index = group_ordinal * per_group + role_offset
        engine_id = str(engine_id_base + instance_index)

        has_kv_port_base = "kvPortBase" in identity
        has_instance_stride = "instanceStride" in identity
        if has_kv_port_base != has_instance_stride:
            raise ResolutionError(
                "identity.kvPortBase and identity.instanceStride must be set together"
            )
        has_port = has_kv_port_base
        replacements: dict[str, Any] = {ENGINE_ID_SENTINEL: engine_id}
        if has_port:
            kv_port_base = _require_int(
                identity, "kvPortBase", minimum=1, maximum=65535
            )
            instance_stride = _require_int(identity, "instanceStride", minimum=1)
            port_span = _require_int(meta, "portSpan", minimum=1)
            minimum_stride = max(100, port_span)
            if instance_stride < minimum_stride:
                raise ResolutionError(
                    f"identity.instanceStride must be >= {minimum_stride} "
                    f"for portSpan={port_span}, got {instance_stride}"
                )
            kv_port = kv_port_base + instance_index * instance_stride
            last_reserved_port = kv_port + port_span - 1
            if kv_port > 65535 or last_reserved_port > 65535:
                raise ResolutionError(
                    "resolved KV-port range exceeds 65535: "
                    f"base={kv_port}, span={port_span}, last={last_reserved_port}"
                )
            replacements[KV_PORT_SENTINEL] = kv_port
        elif "portSpan" in meta:
            raise ResolutionError(
                "metadata.portSpan must be omitted when identity has no KV-port fields"
            )

        _validate_dynamic_shape(template, has_port, role_kind)
        _validate_sentinels(
            template,
            identity_enabled=True,
            has_port=has_port,
        )
        resolved = _replace_exact_values(template, replacements)

    _ensure_no_residual_sentinels(resolved)
    if probe_registry:
        _probe_connector_registry(resolved)
    return resolved


def _read_json_object(path: str, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ResolutionError(f"cannot read {description} {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ResolutionError(f"invalid JSON in {description} {path!r}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResolutionError(f"{description} {path!r} must contain a JSON object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve a vLLM KV-transfer config for the current Pod"
    )
    parser.add_argument(
        "--template",
        default=os.environ.get(
            "UC_KV_TRANSFER_TEMPLATE_PATH",
            "/vllm-workspace/UnifiedCache/entrypoint/kv-transfer.template.json",
        ),
        help=(
            "path to kv-transfer.template.json (default: "
            "UC_KV_TRANSFER_TEMPLATE_PATH or the projected entrypoint path)"
        ),
    )
    parser.add_argument(
        "--meta",
        default=os.environ.get(
            "UC_KV_TRANSFER_META_PATH",
            "/vllm-workspace/UnifiedCache/entrypoint/kv-transfer.meta.json",
        ),
        help=(
            "path to kv-transfer.meta.json (default: UC_KV_TRANSFER_META_PATH "
            "or the projected entrypoint path)"
        ),
    )
    parser.add_argument(
        "--group-name",
        default=os.environ.get("UC_PD_GROUP_NAME"),
        help="ServingGroup label (default: UC_PD_GROUP_NAME)",
    )
    parser.add_argument(
        "--role-id",
        default=os.environ.get("UC_PD_ROLE_ID"),
        help="role replica label (default: UC_PD_ROLE_ID)",
    )
    parser.add_argument(
        "--output",
        help=(
            "write the compact JSON result to this file instead of stdout; "
            "the chart runtime always uses this channel so plugin diagnostics "
            "cannot corrupt command data"
        ),
    )
    parser.add_argument(
        "--skip-registry-probe",
        action="store_true",
        help=(
            "skip vLLM plugin/factory validation; intended only for offline "
            "resolver tests and never set by the chart runtime"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        template = _read_json_object(args.template, "KV-transfer template")
        meta = _read_json_object(args.meta, "KV-transfer metadata")
        resolved = resolve_kv_transfer_config(
            template,
            meta,
            group_name=args.group_name,
            role_id=args.role_id,
            probe_registry=not args.skip_registry_probe,
        )
    except ResolutionError as exc:
        print(f"kv-transfer resolver: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(resolved, separators=(",", ":"), ensure_ascii=False)
    if args.output:
        try:
            Path(args.output).write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(
                f"kv-transfer resolver: cannot write output {args.output!r}: {exc}",
                file=sys.stderr,
            )
            return 2
    else:
        # Kept for direct diagnostics/tests. The chart runtime uses --output.
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
