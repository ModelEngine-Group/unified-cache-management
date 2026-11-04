"""Read and encode release data with deterministic serialization."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


def load_yaml_value(text: str, *, context: str) -> Any:
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except (ValueError, yaml.YAMLError) as error:
        raise ValueError(f"{context}: malformed YAML: {error}") from error


def load_yaml(path: Path) -> dict[str, Any]:
    value = load_yaml_value(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def load_json_value(path: Path) -> Any:

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def load_json(path: Path) -> dict[str, Any]:
    value = load_json_value(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_json_array(path: Path) -> list[Any]:
    value = load_json_value(path)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    return value


def _resolve_ref(root: dict[str, Any], reference: str) -> Any:
    if not reference.startswith("#/"):
        raise ValueError(f"unsupported schema reference: {reference}")
    value: Any = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"unresolved schema reference: {reference}")
        value = value[part]
    return value


def validate_schema(
    instance: Any,
    schema: Any,
    *,
    root: dict[str, Any] | None = None,
    path: str = "$",
) -> None:
    if schema is False:
        raise ValueError(f"{path}: value is forbidden by schema")
    if schema is True:
        return
    if not isinstance(schema, dict):
        raise ValueError(f"{path}: invalid schema node")
    root_schema = root or schema
    if "$ref" in schema:
        validate_schema(
            instance,
            _resolve_ref(root_schema, schema["$ref"]),
            root=root_schema,
            path=path,
        )
    if "oneOf" in schema:
        matches = 0
        errors: list[str] = []
        for option in schema["oneOf"]:
            try:
                validate_schema(instance, option, root=root_schema, path=path)
                matches += 1
            except ValueError as error:
                errors.append(str(error))
        if matches != 1:
            detail = errors[0] if errors else f"matched {matches} branches"
            raise ValueError(f"{path}: oneOf requires exactly one match: {detail}")
    expected_type = schema.get("type")
    if expected_type is not None:
        type_checks = {
            "object": lambda value: isinstance(value, dict),
            "array": lambda value: isinstance(value, list),
            "string": lambda value: isinstance(value, str),
            "integer": lambda value: isinstance(value, int)
            and (not isinstance(value, bool)),
            "boolean": lambda value: isinstance(value, bool),
            "number": lambda value: isinstance(value, (int, float))
            and (not isinstance(value, bool)),
            "null": lambda value: value is None,
        }  # noqa: E501
        if expected_type not in type_checks or not type_checks[expected_type](instance):
            raise ValueError(f"{path}: expected {expected_type}")
    if "const" in schema and instance != schema["const"]:
        raise ValueError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise ValueError(f"{path}: expected one of {schema['enum']!r}")
    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            raise ValueError(f"{path}: string is shorter than minLength")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            raise ValueError(
                f"{path}: value does not match pattern {schema['pattern']!r}"
            )  # noqa: E501
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise ValueError(f"{path}: value is below minimum {schema['minimum']}")
    if isinstance(instance, dict):
        if len(instance) < schema.get("minProperties", 0):
            raise ValueError(f"{path}: object has fewer than minProperties")
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            raise ValueError(f"{path}: object has more than maxProperties")
        missing = [key for key in schema.get("required", []) if key not in instance]
        if missing:
            raise ValueError(f"{path}: missing required properties {missing}")
        properties = schema.get("properties", {})
        property_names = schema.get("propertyNames")
        if property_names is not None:
            for key in instance:
                validate_schema(
                    key, property_names, root=root_schema, path=f"{path}.<property>"
                )  # noqa: E501
        if schema.get("additionalProperties") is False:
            extras = sorted(set(instance) - set(properties))
            if extras:
                raise ValueError(
                    f"Additional properties are not allowed at {path}: {extras}"
                )
        for key, value in instance.items():
            if key in properties:
                validate_schema(
                    value, properties[key], root=root_schema, path=f"{path}.{key}"
                )  # noqa: E501
            elif isinstance(schema.get("additionalProperties"), dict):
                validate_schema(
                    value,
                    schema["additionalProperties"],
                    root=root_schema,
                    path=f"{path}.{key}",
                )  # noqa: E501
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            raise ValueError(f"{path}: array is shorter than minItems")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise ValueError(f"{path}: array is longer than maxItems")
        if schema.get("uniqueItems"):
            encoded = [canonical_bytes(item) for item in instance]
            if len(encoded) != len(set(encoded)):
                raise ValueError(f"{path}: array items must be unique")
        prefix_items = schema.get("prefixItems", [])
        for index, item_schema in enumerate(prefix_items):
            if index < len(instance):
                validate_schema(
                    instance[index],
                    item_schema,
                    root=root_schema,
                    path=f"{path}[{index}]",
                )
        item_schema = schema.get("items")
        if item_schema is not None:
            start = len(prefix_items) if prefix_items else 0
            for index in range(start, len(instance)):
                validate_schema(
                    instance[index],
                    item_schema,
                    root=root_schema,
                    path=f"{path}[{index}]",
                )  # noqa: E501


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()  # noqa: E501


def sha256_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)
