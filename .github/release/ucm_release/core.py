# fmt: off
from __future__ import annotations

import copy
import hashlib
import json
import os
import posixpath
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = RELEASE_ROOT / "release.yaml"
DEFAULT_SCHEMA_DIR = RELEASE_ROOT / "schemas"
OCI_REPOSITORY_PATTERN = re.compile('^[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?(?:/[a-z0-9]+(?:(?:[._]|-+)[a-z0-9]+)*)+$')  # fmt: skip  # noqa: E501
OCI_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class CpuToolchainAuthority:

    cpu_arch: str
    oci_platform: str
    wheel_arch: str
    elf_machine: int
    elf_machine_name: str
    host_machine_aliases: tuple[str, ...]


CPU_TOOLCHAIN_AUTHORITIES: Mapping[str, CpuToolchainAuthority] = MappingProxyType({'amd64': CpuToolchainAuthority(cpu_arch='amd64', oci_platform='linux/amd64', wheel_arch='x86_64', elf_machine=62, elf_machine_name='EM_X86_64', host_machine_aliases=('x86_64', 'amd64')), 'arm64': CpuToolchainAuthority(cpu_arch='arm64', oci_platform='linux/arm64', wheel_arch='aarch64', elf_machine=183, elf_machine_name='EM_AARCH64', host_machine_aliases=('aarch64', 'arm64'))})  # fmt: skip  # noqa: E501


def cpu_toolchain_authority(
    cpu_arch: object, *, location: str = "CPU/tool architecture"
) -> CpuToolchainAuthority:
    authority = CPU_TOOLCHAIN_AUTHORITIES.get(cpu_arch) if isinstance(cpu_arch, str) else None  # fmt: skip  # noqa: E501
    if authority is None: raise ValueError(f'unsupported CPU arch {cpu_arch!r} at {location}')  # noqa: E701,E501
    return authority


def host_cpu_toolchain_authority(host_machine: object) -> CpuToolchainAuthority:
    normalized = host_machine.lower() if isinstance(host_machine, str) else None
    matches = [authority for authority in CPU_TOOLCHAIN_AUTHORITIES.values() if normalized in authority.host_machine_aliases]  # fmt: skip  # noqa: E501
    if len(matches) != 1: raise ValueError(f'unsupported CPU/tool host architecture: {host_machine!r}')  # noqa: E701,E501
    return matches[0]


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping: raise ValueError(f'duplicate YAML key: {key}')  # noqa: E701
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)  # fmt: skip  # noqa: E501


def load_yaml_value(text: str, *, context: str) -> Any:
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except (ValueError, yaml.YAMLError) as error:
        raise ValueError(f'{context}: malformed YAML: {error}') from error


def load_yaml(path: Path) -> dict[str, Any]:
    value = load_yaml_value(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(value, dict): raise ValueError(f'{path} must contain a mapping')  # noqa: E701,E501
    return value


def load_json_value(path: Path) -> Any:

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result: raise ValueError(f'duplicate JSON key {key!r} in {path}')  # noqa: E701,E501
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def load_json(path: Path) -> dict[str, Any]:
    value = load_json_value(path)
    if not isinstance(value, dict): raise ValueError(f'{path} must contain a JSON object')  # noqa: E701,E501
    return value


def load_json_array(path: Path) -> list[Any]:
    value = load_json_value(path)
    if not isinstance(value, list): raise ValueError(f'{path} must contain a JSON array')  # noqa: E701,E501
    return value


def _resolve_ref(root: dict[str, Any], reference: str) -> Any:
    if not reference.startswith('#/'): raise ValueError(f'unsupported schema reference: {reference}')  # noqa: E701,E501
    value: Any = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or part not in value: raise ValueError(f'unresolved schema reference: {reference}')  # noqa: E701,E501
        value = value[part]
    return value


def validate_schema(
    instance: Any,
    schema: Any,
    *,
    root: dict[str, Any] | None = None,
    path: str = "$",
) -> None:
    if schema is False: raise ValueError(f'{path}: value is forbidden by schema')  # noqa: E701,E501
    if schema is True:
        return
    if not isinstance(schema, dict): raise ValueError(f'{path}: invalid schema node')  # noqa: E701,E501
    root_schema = root or schema
    if '$ref' in schema: validate_schema(instance, _resolve_ref(root_schema, schema['$ref']), root=root_schema, path=path)  # noqa: E701,E501
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
        type_checks = {'object': lambda value: isinstance(value, dict), 'array': lambda value: isinstance(value, list), 'string': lambda value: isinstance(value, str), 'integer': lambda value: isinstance(value, int) and (not isinstance(value, bool)), 'boolean': lambda value: isinstance(value, bool), 'number': lambda value: isinstance(value, (int, float)) and (not isinstance(value, bool)), 'null': lambda value: value is None}  # fmt: skip  # noqa: E501
        if expected_type not in type_checks or not type_checks[expected_type](instance):
            raise ValueError(f"{path}: expected {expected_type}")
    if 'const' in schema and instance != schema['const']: raise ValueError(f"{path}: expected constant {schema['const']!r}")  # noqa: E701,E501
    if 'enum' in schema and instance not in schema['enum']: raise ValueError(f"{path}: expected one of {schema['enum']!r}")  # noqa: E701,E501
    if isinstance(instance, str):
        if len(instance) < schema.get('minLength', 0): raise ValueError(f'{path}: string is shorter than minLength')  # noqa: E701,E501
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            raise ValueError(f"{path}: value does not match pattern {schema['pattern']!r}")  # fmt: skip  # noqa: E501
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise ValueError(f"{path}: value is below minimum {schema['minimum']}")
    if isinstance(instance, dict):
        if len(instance) < schema.get('minProperties', 0): raise ValueError(f'{path}: object has fewer than minProperties')  # noqa: E701,E501
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            raise ValueError(f"{path}: object has more than maxProperties")
        missing = [key for key in schema.get("required", []) if key not in instance]
        if missing: raise ValueError(f'{path}: missing required properties {missing}')  # noqa: E701,E501
        properties = schema.get("properties", {})
        property_names = schema.get("propertyNames")
        if property_names is not None:
            for key in instance:
                validate_schema(key, property_names, root=root_schema, path=f'{path}.<property>')  # fmt: skip  # noqa: E501
        if schema.get("additionalProperties") is False:
            extras = sorted(set(instance) - set(properties))
            if extras: raise ValueError(f'Additional properties are not allowed at {path}: {extras}')  # noqa: E701,E501
        for key, value in instance.items():
            if key in properties:
                validate_schema(value, properties[key], root=root_schema, path=f'{path}.{key}')  # fmt: skip  # noqa: E501
            elif isinstance(schema.get("additionalProperties"), dict):
                validate_schema(value, schema['additionalProperties'], root=root_schema, path=f'{path}.{key}')  # fmt: skip  # noqa: E501
    if isinstance(instance, list):
        if len(instance) < schema.get('minItems', 0): raise ValueError(f'{path}: array is shorter than minItems')  # noqa: E701,E501
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise ValueError(f"{path}: array is longer than maxItems")
        if schema.get("uniqueItems"):
            encoded = [canonical_bytes(item) for item in instance]
            if len(encoded) != len(set(encoded)): raise ValueError(f'{path}: array items must be unique')  # noqa: E701,E501
        prefix_items = schema.get("prefixItems", [])
        for index, item_schema in enumerate(prefix_items):
            if index < len(instance): validate_schema(instance[index], item_schema, root=root_schema, path=f'{path}[{index}]')  # noqa: E701,E501
        item_schema = schema.get("items")
        if item_schema is not None:
            start = len(prefix_items) if prefix_items else 0
            for index in range(start, len(instance)):
                validate_schema(instance[index], item_schema, root=root_schema, path=f'{path}[{index}]')  # fmt: skip  # noqa: E501


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()  # fmt: skip  # noqa: E501


def sha256_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_version(path: Path | None = None) -> str:
    version_path = path or (REPO_ROOT / "version.ini")
    for line in version_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key == "VLLM_UC_VERSION" and value:
            return value
    raise ValueError(f"VLLM_UC_VERSION is missing from {version_path}")


def _pep440_public(version: str) -> str:
    base = version.split('+', 1)[0]
    if '.dev' in base:
        base = re.split(r'\.dev\d+', base)[0]
    return base


def _oci_tag_version(version: str) -> str:
    return version.replace('+', '.')  # fmt: skip  # noqa: E501


def derive_chart_version(version: str) -> str:
    parsed = _pep440_version(version, "UCM release version")
    if parsed.dev is not None:
        if parsed.pre is not None or parsed.post is not None or parsed.local is not None:
            raise ValueError(
                f"unsupported UCM draft version for Chart SemVer: {version}"
            )
        return f"{parsed.base_version}-draft.{parsed.dev}"
    public = _pep440_public(version)
    match = re.fullmatch(r"([0-9]+\.[0-9]+\.[0-9]+)rc([0-9]+)", public)
    if match is None:
        if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", public): return public  # fmt: skip  # noqa: E701,E501
        raise ValueError(f'unsupported UCM release version for Chart SemVer: {version}')  # noqa: E701,E501
    return f"{match.group(1)}-rc.{match.group(2)}"


def _pep440_version(value: object, location: str) -> Version:
    try:
        return Version(str(value))
    except InvalidVersion as error:
        raise ValueError(f'{location} must be a valid PEP 440 version: {value!r}') from error  # fmt: skip  # noqa: E501


def _pep440_specifier(value: object, location: str) -> SpecifierSet:
    try:
        return SpecifierSet(str(value))
    except InvalidSpecifier as error:
        raise ValueError(f'{location} must be a valid PEP 440 specifier: {value!r}') from error  # fmt: skip  # noqa: E501


def _specifier_interval(
    value: object,
) -> tuple[Version | None, bool, Version | None, bool] | None:
    specifiers = _pep440_specifier(value, "compatibility version selector")
    lower: Version | None = None
    lower_inclusive = False
    upper: Version | None = None
    upper_inclusive = False

    def add_lower(candidate: Version, inclusive: bool) -> None:
        nonlocal lower, lower_inclusive
        if lower is None or candidate > lower:
            lower, lower_inclusive = candidate, inclusive
        if candidate == lower: lower_inclusive = lower_inclusive and inclusive  # noqa: E701,E501

    def add_upper(candidate: Version, inclusive: bool) -> None:
        nonlocal upper, upper_inclusive
        if upper is None or candidate < upper:
            upper, upper_inclusive = candidate, inclusive
        if candidate == upper: upper_inclusive = upper_inclusive and inclusive  # noqa: E701,E501

    for specifier in specifiers:
        operator = specifier.operator
        raw_version = specifier.version
        if operator == "!=":
            # Exclusions can only shrink the interval. Ignoring them is conservative:
            # an uncertain pair is rejected instead of accepting future ambiguity.
            continue
        if operator == "==" and raw_version.endswith(".*"):
            prefix = _pep440_version(raw_version.removesuffix('.*'), 'compatibility wildcard prefix')  # fmt: skip  # noqa: E501
            if (
                prefix.epoch != 0
                or prefix.pre
                or prefix.post
                or prefix.dev
                or prefix.local
            ):
                return None
            release = prefix.release
            upper_release = (*release[:-1], release[-1] + 1)
            add_lower(Version(".".join(map(str, release)) + ".dev0"), True)
            add_upper(Version(".".join(map(str, upper_release))), False)
            continue
        if operator == "~=":
            compatible = _pep440_version(raw_version, 'compatibility compatible-release selector')  # fmt: skip  # noqa: E501
            if (
                compatible.epoch != 0
                or compatible.local is not None
                or len(compatible.release) < 2
            ):
                return None
            upper_prefix = compatible.release[:-1]
            upper_release = (*upper_prefix[:-1], upper_prefix[-1] + 1)
            add_lower(compatible, True)
            add_upper(Version(".".join(map(str, upper_release))), False)
            continue
        if operator not in {"==", ">=", ">", "<=", "<"}:
            return None
        version = _pep440_version(raw_version, "compatibility version boundary")
        if operator == "==":
            add_lower(version, True)
            add_upper(version, True)
        elif operator == ">=":
            add_lower(version, True)
        elif operator == ">":
            add_lower(version, False)
        elif operator == "<=":
            add_upper(version, True)
        else:
            add_upper(version, False)
    return lower, lower_inclusive, upper, upper_inclusive


def _version_specifiers_may_overlap(left: object, right: object) -> bool:
    left_specifiers = _pep440_specifier(left, "left compatibility version selector")
    right_specifiers = _pep440_specifier(right, "right compatibility version selector")
    for specifiers in (left_specifiers, right_specifiers):
        for specifier in specifiers:
            if specifier.operator != "==" or specifier.version.endswith(".*"):
                continue
            witness = _pep440_version(specifier.version, 'compatibility exact-version witness')  # fmt: skip  # noqa: E501
            if left_specifiers.contains(
                witness, prereleases=True
            ) and right_specifiers.contains(witness, prereleases=True):
                return True
    combined = ",".join(part for part in (str(left), str(right)) if part)
    interval = _specifier_interval(combined)
    if interval is None:
        return True
    lower, lower_inclusive, upper, upper_inclusive = interval
    if lower is None or upper is None or lower < upper:
        return True
    if lower > upper:
        return False
    return lower_inclusive and upper_inclusive


def _compatibility_rules_semantically_overlap(
    left: dict[str, Any],
    right: dict[str, Any],
    products_by_id: dict[str, dict[str, Any]],
) -> bool:
    common_products = set(left["upstream_products"]) & set(right["upstream_products"])
    if not common_products or left["accelerator"] != right["accelerator"]:
        return False
    common_variants = set(left["variants"]) & set(right["variants"])
    if not any(
        common_variants
        & {variant["id"] for variant in products_by_id[product_id]["variants"]}
        for product_id in common_products
    ):
        return False
    for field in (
        "accelerator_runtimes",
        "npu_architectures",
        "operating_systems",
        "cpu_architectures",
        "python_abis",
        "upstream_channels",
    ):
        if not (set(left[field]) & set(right[field])):
            return False
    return _version_specifiers_may_overlap(left['version_specifier'], right['version_specifier'])  # fmt: skip  # noqa: E501


def _require_unique_ids(items: list[dict[str, Any]], label: str) -> None:
    seen: set[str] = set()
    for item in items:
        identifier = item["id"]
        if identifier in seen: raise ValueError(f'duplicate {label} id {identifier!r}')  # noqa: E701,E501
        seen.add(identifier)


_BUILDER_CHECK_FIELDS = {'python': {'kind', 'version', 'abi'}, 'python-soabi': {'kind', 'prefix'}, 'command': {'kind', 'name'}, 'command-version': {'kind', 'name', 'arguments', 'contains'}, 'file': {'kind', 'path'}, 'directory': {'kind', 'path'}, 'library-cache': {'kind', 'path'}, 'shared-library-dependencies': {'kind', 'path'}}  # fmt: skip  # noqa: E501


def _validate_builder_checks(
    checks: object, *, profile: dict[str, Any], location: str
) -> None:
    if not isinstance(checks, list) or not checks: raise ValueError(f'{location} builder checks must be a non-empty array')  # noqa: E701,E501
    seen: set[bytes] = set()
    requirements: dict[tuple[str, str], tuple[str, bytes]] = {}
    python_checks = 0
    soabi_checks = 0
    for index, check in enumerate(checks):
        check_location = f"{location}.checks[{index}]"
        if not isinstance(check, dict): raise ValueError(f'{check_location} must be an object')  # noqa: E701,E501
        kind = check.get("kind")
        fields = _BUILDER_CHECK_FIELDS.get(str(kind))
        if fields is None: raise ValueError(f'unsupported builder check kind {kind!r}')  # noqa: E701,E501
        if set(check) != fields: raise ValueError(f'{check_location} fields are not exact')  # noqa: E701,E501
        encoded = canonical_bytes(check)
        if encoded in seen: raise ValueError(f'duplicate builder check at {check_location}')  # noqa: E701,E501
        seen.add(encoded)
        if kind == "python":
            python_checks += 1
            version = check["version"]
            abi = check["abi"]
            if (
                not isinstance(version, str)
                or re.fullmatch(r"[0-9]+\.[0-9]+", version) is None
                or not isinstance(abi, str)
                or abi != "cp" + version.replace(".", "")
                or version != profile["python_version"]
                or abi != profile["python_abi"]
            ):
                raise ValueError(f'{check_location} differs from profile Python authority')  # fmt: skip  # noqa: E501
            target = ("python", "runtime")
        elif kind == "python-soabi":
            soabi_checks += 1
            if (
                not isinstance(check["prefix"], str)
                or re.fullmatch(r"[a-z][a-z0-9_-]*", check["prefix"]) is None
            ):
                raise ValueError(f"{check_location}.prefix is invalid")
            target = ("python-soabi", "runtime")
        elif kind in {"command", "command-version"}:
            name = check["name"]
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", name) is None
            ):
                raise ValueError(f"{check_location}.name is invalid")
            if kind == "command-version":
                arguments = check["arguments"]
                contains = check["contains"]
                if (
                    not isinstance(arguments, list)
                    or not arguments
                    or not all(
                        isinstance(argument, str)
                        and argument
                        and "\x00" not in argument
                        for argument in arguments
                    )
                    or not isinstance(contains, str)
                    or not contains
                    or "\x00" in contains
                ):
                    raise ValueError(f"{check_location} arguments are invalid")
            target = ("command", name)
        else:
            path = check["path"]
            if (
                not isinstance(path, str)
                or not path.startswith("/")
                or any(character.isspace() for character in path)
                or "\x00" in path
            ):
                raise ValueError(f"{check_location}.path is invalid")
            normalized_path = posixpath.normpath("/" + path.lstrip("/"))
            target = ('filesystem-node' if kind in {'file', 'directory'} else str(kind), normalized_path)  # fmt: skip  # noqa: E501
        previous = requirements.get(target)
        if previous is not None:
            previous_kind, previous_encoded = previous
            if target[0] == "filesystem-node" and previous_kind == kind:
                raise ValueError(f"duplicate builder check at {check_location}")
            if previous_encoded != encoded: raise ValueError(f'conflicting builder checks for {target[1]!r}')  # noqa: E701,E501
        requirements[target] = (str(kind), encoded)
    if python_checks != 1 or soabi_checks != 1:
        raise ValueError(f'{location} requires exactly one Python and one Python SOABI check')  # fmt: skip  # noqa: E501


def _validate_builder_root(root: object, *, location: str) -> None:
    if root is None:
        return
    if not isinstance(root, dict):
        raise ValueError(f"{location}.root must be an object")
    unresolved_fields = {"repository", "tag"}
    resolved_fields = unresolved_fields | {
        "index_digest",
        "manifest_digest",
        "config_digest",
    }
    if frozenset(root) not in {
        frozenset(unresolved_fields),
        frozenset(resolved_fields),
    }:
        raise ValueError(f"{location}.root must be unresolved or fully resolved")
    if OCI_REPOSITORY_PATTERN.fullmatch(str(root.get("repository"))) is None:
        raise ValueError(f"{location}.root.repository must use canonical OCI syntax")
    if OCI_TAG_PATTERN.fullmatch(str(root.get("tag"))) is None:
        raise ValueError(f"{location}.root.tag must use canonical OCI tag syntax")
    for digest_name in sorted(resolved_fields - unresolved_fields):
        if digest_name in root and re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(root[digest_name])
        ) is None:
            raise ValueError(
                f"{location}.root.{digest_name} must be an exact sha256 digest"
            )


def _validate_resolved_builder_roots(catalog: dict[str, Any]) -> None:
    required = {
        "repository",
        "tag",
        "index_digest",
        "manifest_digest",
        "config_digest",
    }
    for profile_index, profile in enumerate(catalog.get("wheel_profiles", [])):
        for architecture, builder in profile.get("builders", {}).items():
            location = f"wheel_profiles[{profile_index}].builders.{architecture}"
            root = builder.get("root") if isinstance(builder, dict) else None
            if not isinstance(root, dict) or set(root) != required:
                raise ValueError(f"{location} builder root must be resolved")


def _exact_runtime_requirement(value: object) -> tuple[str, str, str]:
    if not isinstance(value, str) or not value: raise ValueError('python runtime requirement is invalid')  # noqa: E701,E501
    try:
        requirement = Requirement(value)
    except InvalidRequirement as error:
        raise ValueError("python runtime requirement is invalid") from error
    specifiers = list(requirement.specifier)
    if (
        requirement.url is not None
        or requirement.marker is not None
        or requirement.extras
        or len(specifiers) != 1
        or specifiers[0].operator != "=="
        or "*" in specifiers[0].version
    ):
        raise ValueError("python runtime requirement must be one exact version")
    try:
        version = str(Version(specifiers[0].version))
    except InvalidVersion as error:
        raise ValueError("python runtime requirement version is invalid") from error
    name = canonicalize_name(requirement.name)
    return name, version, f"{name}=={version}"


def python_runtime_requirements(catalog: dict[str, Any]) -> list[str]:
    direct_requirements = catalog.get("wheel_runtime_requirements")
    if direct_requirements is not None:
        if (
            not isinstance(direct_requirements, list)
            or not direct_requirements
            or not all(isinstance(item, str) for item in direct_requirements)
        ):
            raise ValueError("Wheel runtime requirements must be a non-empty string list")
        resolved = [_exact_runtime_requirement(item)[2] for item in direct_requirements]
        if len(set(resolved)) != len(resolved):
            raise ValueError("Wheel runtime requirements contain duplicate packages")
        return sorted(resolved)
    declarations = catalog.get("python_runtime_dependencies")
    if not declarations:
        return []
    if (
        not isinstance(declarations, list)
        or not 1 <= len(declarations) <= 64
        or any(not isinstance(item, dict) for item in declarations)
    ):
        raise ValueError("python runtime dependency declarations are invalid")
    build_lock = catalog.get("python_build_lock")
    packages = build_lock.get("packages") if isinstance(build_lock, dict) else None
    resolved: list[str] = []
    identities: set[str] = set()
    for declaration in declarations:
        import_name = declaration.get("import_name")
        if (
            not isinstance(import_name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", import_name) is None
        ):
            raise ValueError("python runtime dependency import name is invalid")
        if set(declaration) == {"requirement", "import_name", "wheel_artifacts"}:
            identity, _, resolved_requirement = _exact_runtime_requirement(declaration['requirement'])  # fmt: skip  # noqa: E501
            if not isinstance(declaration['wheel_artifacts'], dict): raise ValueError('python runtime wheel artifacts are invalid')  # noqa: E701,E501
        elif set(declaration) == {"python_build_lock", "import_name"}:
            package_name = declaration["python_build_lock"]
            package = packages.get(package_name) if isinstance(packages, dict) and isinstance(package_name, str) else None  # fmt: skip  # noqa: E501
            if (
                not isinstance(package_name, str)
                or not package_name
                or not isinstance(package, dict)
                or set(package) != {"version", "filename", "sha256"}
                or not isinstance(package["version"], str)
                or not package["version"]
                or not isinstance(package["filename"], str)
                or not package["filename"].endswith(".whl")
                or re.fullmatch(r"sha256:[0-9a-f]{64}", str(package["sha256"])) is None
            ):
                raise ValueError(f'python build lock {package_name!r} runtime authority is invalid')  # fmt: skip  # noqa: E501
            identity = canonicalize_name(package_name)
            resolved_requirement = f"{identity}=={package['version']}"
        else:
            raise ValueError("python runtime dependency declaration is invalid")
        if identity in identities: raise ValueError(f'duplicate python runtime dependency {identity!r}')  # noqa: E701,E501
        identities.add(identity)
        resolved.append(resolved_requirement)
    return sorted(resolved)


def runtime_dependency_records(
    catalog: dict[str, Any], python_abi: str, architecture: str
) -> list[dict[str, str]]:
    requirements = set(python_runtime_requirements(catalog))
    build_lock = catalog["python_build_lock"]["packages"]
    records: list[dict[str, str]] = []
    for declaration in catalog["python_runtime_dependencies"]:
        if "requirement" in declaration:
            name, version, requirement = _exact_runtime_requirement(declaration['requirement'])  # fmt: skip  # noqa: E501
            wheel = python_abi_artifact(declaration['wheel_artifacts'], python_abi, architecture, label=f'python_runtime_dependencies.{name}.wheel_artifacts')  # fmt: skip  # noqa: E501
        else:
            name = canonicalize_name(declaration["python_build_lock"])
            package = build_lock[declaration["python_build_lock"]]
            version = package["version"]
            requirement = f"{name}=={version}"
            wheel = {"filename": package["filename"], "sha256": package["sha256"]}
        records.append({'name': name, 'version': version, 'requirement': requirement, 'import_name': declaration['import_name'], 'filename': wheel['filename'], 'sha256': wheel['sha256']})  # fmt: skip  # noqa: E501
    records.sort(key=lambda item: item["name"])
    if (
        {item["requirement"] for item in records} != requirements
        or len({item["name"] for item in records}) != len(records)
        or len({item["filename"] for item in records}) != len(records)
    ):
        raise ValueError("python runtime dependency wheel authority is ambiguous")
    return records


def build_tool_dependency_records(
    catalog: dict[str, Any], python_abi: str, architecture: str
) -> list[dict[str, str]]:
    direct_requirements = catalog.get("wheel_build_requirements")
    if direct_requirements is not None:
        if (
            not isinstance(direct_requirements, list)
            or not direct_requirements
            or not all(isinstance(item, str) for item in direct_requirements)
        ):
            raise ValueError("Wheel build requirements must be a non-empty string list")
        records = []
        for item in direct_requirements:
            name, version, requirement = _exact_runtime_requirement(item)
            records.append(
                {"name": name, "version": version, "requirement": requirement}
            )
        records.sort(key=lambda item: item["name"])
        if len({record["name"] for record in records}) != len(records):
            raise ValueError("Wheel build requirements contain duplicate packages")
        return records
    build_lock = catalog.get("python_build_lock")
    packages = build_lock.get("packages") if isinstance(build_lock, dict) else None
    if not isinstance(packages, dict) or not 1 <= len(packages) <= 64:
        raise ValueError("python build tool package authority is invalid")
    records: list[dict[str, str]] = []
    for package_name, package in packages.items():
        if not isinstance(package, dict): raise ValueError('python build tool package authority is invalid')  # noqa: E701,E501
        name = canonicalize_name(package_name)
        records.append({'name': name, 'version': str(package['version']), 'requirement': f"{name}=={package['version']}", 'filename': package['filename'], 'sha256': package['sha256']})  # fmt: skip  # noqa: E501
    for name, lock_record, artifact in (
        (
            "pyyaml",
            build_lock.get("pyyaml"),
            python_abi_artifact(
                build_lock.get("pyyaml", {}).get("artifacts"),
                python_abi,
                architecture,
                label="python_build_lock.pyyaml.artifacts",
            ),
        ),
        (
            "cmake",
            build_lock.get("cmake"),
            (
                build_lock.get("cmake", {}).get("artifacts", {}).get(architecture)
                if isinstance(build_lock.get("cmake", {}).get("artifacts"), dict)
                else None
            ),
        ),
    ):
        if (
            not isinstance(lock_record, dict)
            or not isinstance(lock_record.get("version"), str)
            or not lock_record["version"]
            or not isinstance(artifact, dict)
            or set(artifact) != {"filename", "sha256"}
            or not isinstance(artifact["filename"], str)
            or not artifact["filename"].endswith(".whl")
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(artifact["sha256"])) is None
        ):
            raise ValueError(f"python build tool {name!r} has no exact task artifact")
        records.append({'name': name, 'version': lock_record['version'], 'requirement': f"{name}=={lock_record['version']}", 'filename': artifact['filename'], 'sha256': artifact['sha256']})  # fmt: skip  # noqa: E501
    records.sort(key=lambda item: item["name"])
    if (
        len(records) > 64
        or len({record["name"] for record in records}) != len(records)
        or len({record["filename"] for record in records}) != len(records)
    ):
        raise ValueError("python build tool wheel authority is ambiguous")
    return records


def python_abi_artifact(
    artifacts: object, python_abi: str, architecture: str, *, label: str
) -> dict[str, str]:
    abi_artifacts = artifacts.get(python_abi) if isinstance(artifacts, dict) else None
    artifact = abi_artifacts.get(architecture) if isinstance(abi_artifacts, dict) else None  # fmt: skip  # noqa: E501
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"filename", "sha256"}
        or not isinstance(artifact["filename"], str)
        or not artifact["filename"]
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(artifact["sha256"])) is None
    ):
        raise ValueError(f"{label} has no exact {python_abi}/{architecture} artifact")
    return copy.deepcopy(artifact)


def _validate_catalog_cpu_toolchains(catalog: dict[str, Any]) -> None:
    declarations: list[tuple[str, object]] = []
    runner_map = catalog.get("runner_map")
    if isinstance(runner_map, dict): declarations.extend(((f'runner_map.{key}', key) for key in runner_map))  # noqa: E701,E501
    for index, profile in enumerate(catalog.get("wheel_profiles", [])):
        if not isinstance(profile, dict):
            continue
        architectures = profile.get("cpu_arch")
        if isinstance(architectures, list):
            declarations.extend(((f'wheel_profiles[{index}].cpu_arch', architecture) for architecture in architectures))  # fmt: skip  # noqa: E501
        builders = profile.get("builders")
        if isinstance(builders, dict):
            declarations.extend(((f'wheel_profiles[{index}].builders.{architecture}', architecture) for architecture in builders))  # fmt: skip  # noqa: E501
    for index, product in enumerate(catalog.get("upstream_products", [])):
        if not isinstance(product, dict):
            continue
        architectures = product.get("required_cpu_architectures")
        if isinstance(architectures, list):
            declarations.extend(((f'upstream_products[{index}].required_cpu_architectures', architecture) for architecture in architectures))  # fmt: skip  # noqa: E501
    compatibility = catalog.get("compatibility")
    rules = compatibility.get("rules", []) if isinstance(compatibility, dict) else []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        architectures = rule.get("cpu_architectures")
        if isinstance(architectures, list):
            declarations.extend(((f'compatibility.rules[{index}].cpu_architectures', architecture) for architecture in architectures))  # fmt: skip  # noqa: E501
    for location, architecture in declarations:
        cpu_toolchain_authority(architecture, location=location)


def validate_catalog(
    catalog: dict[str, Any], *, repository_root: Path = REPO_ROOT
) -> None:
    _validate_catalog_cpu_toolchains(catalog)
    _pep440_version(catalog.get("ucm_version"), "ucm_version")
    python_runtime_requirements(catalog)
    profiles = catalog.get("wheel_profiles", [])
    products = catalog.get("upstream_products", [])
    if not profiles:
        _require_unique_ids(products, "upstream product")
        for index, product in enumerate(products):
            _pep440_specifier(
                product.get("version_specifier"),
                f"upstream_products[{index}].version_specifier",
            )
            for field in (
                "runtime_repository",
                "target_repository",
                "integration_python_abi",
            ):
                if not isinstance(product.get(field), str) or not product[field]:
                    raise ValueError(
                        f"upstream_products[{index}].{field} must be non-empty"
                    )
        contracts = catalog.get("backend_contracts")
        if not isinstance(contracts, dict) or not contracts:
            raise ValueError("backend_contracts must be a non-empty mapping")
        for backend, contract in contracts.items():
            if not isinstance(backend, str) or not isinstance(contract, dict):
                raise ValueError("backend contract entries are malformed")
            for field in (
                "platform_arg",
                "required_native",
                "forbidden_native",
                "allowed_dt_needed",
                "external_required_dependencies",
            ):
                if field not in contract:
                    raise ValueError(f"backend contract {backend!r} missing {field}")
            if not ({"distribution", "distribution_prefix"} & set(contract)):
                raise ValueError(
                    f"backend contract {backend!r} requires a distribution rule"
                )
        return
    compatibility = catalog.get("compatibility", {})
    rules = compatibility.get("rules", [])
    _require_unique_ids(profiles, "wheel profile")
    _require_unique_ids(products, "upstream product")
    _require_unique_ids(rules, "compatibility rule")
    for index, profile in enumerate(profiles):
        _pep440_version(profile.get('wheel_version'), f'wheel_profiles[{index}].wheel_version')  # fmt: skip  # noqa: E501
        for architecture in profile.get("cpu_arch", []):
            runtime_dependency_records(catalog, profile.get("python_abi"), architecture)
            build_tool_dependency_records(catalog, profile.get('python_abi'), architecture)  # fmt: skip  # noqa: E501
        builders = profile.get("builders")
        if not isinstance(builders, dict): raise ValueError(f'wheel_profiles[{index}].builders must be an object')  # noqa: E701,E501
        for architecture, builder in builders.items():
            if not isinstance(builder, dict): raise ValueError(f'wheel_profiles[{index}].builders.{architecture} must be an object')  # noqa: E701,E501
            _validate_builder_checks(builder.get('checks'), profile=profile, location=f'wheel_profiles[{index}].builders.{architecture}')  # fmt: skip  # noqa: E501
            _validate_builder_root(builder.get('root'), location=f'wheel_profiles[{index}].builders.{architecture}')  # fmt: skip  # noqa: E501
    for index, product in enumerate(products):
        _pep440_specifier(product.get('version_specifier'), f'upstream_products[{index}].version_specifier')  # fmt: skip  # noqa: E501
        _require_unique_ids(product["variants"], "upstream variant")
    products_by_id = {product["id"]: product for product in products}
    for index, rule in enumerate(rules):
        _pep440_specifier(rule.get('version_specifier'), f'compatibility.rules[{index}].version_specifier')  # fmt: skip  # noqa: E501
        referenced_products: list[dict[str, Any]] = []
        for product_id in rule["upstream_products"]:
            product = products_by_id.get(product_id)
            if product is None: raise ValueError(f'unknown upstream product {product_id!r}')  # noqa: E701,E501
            referenced_products.append(product)
        declared_variants = {variant['id'] for product in referenced_products for variant in product['variants']}  # fmt: skip  # noqa: E501
        for variant_id in rule["variants"]:
            if variant_id not in declared_variants: raise ValueError(f'unknown variant {variant_id!r}')  # noqa: E701,E501
    for left_index, left in enumerate(rules):
        for right in rules[left_index + 1 :]:
            if _compatibility_rules_semantically_overlap(left, right, products_by_id):
                raise ValueError(f"compatibility rules have semantic selector overlap: {left['id']!r} and {right['id']!r}")  # fmt: skip  # noqa: E501


def validate_resolved_upstreams(resolved_upstreams: object, *, relaxed: bool = False) -> None:
    if not isinstance(resolved_upstreams, list): raise ValueError('resolved_upstreams must be an array')  # noqa: E701,E501
    snapshot_keys = {'product_id', 'repository', 'tag', 'version', 'channel', 'variant', 'index_digest', 'members', 'target_repository', 'target_tag'}  # fmt: skip  # noqa: E501
    member_keys = {"manifest_digest", "config_digest"}
    digest_pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
    logical_identities: set[tuple[str, ...]] = set()
    for index, snapshot in enumerate(resolved_upstreams):
        location = f"resolved_upstreams[{index}]"
        if not isinstance(snapshot, dict): raise ValueError(f'{location} must be an object')  # noqa: E701,E501
        missing = sorted(snapshot_keys - set(snapshot))
        extras = sorted(set(snapshot) - snapshot_keys)
        if missing or extras: raise ValueError(f'{location} requires exact key set; missing={missing}, extra={extras}')  # noqa: E701,E501
        for key in snapshot_keys - {"members", "index_digest"}:
            if not isinstance(snapshot[key], str) or not snapshot[key]:
                raise ValueError(f"{location}.{key} must be a non-empty string")
        if snapshot['channel'] not in {'stable', 'rc'}: raise ValueError(f'{location}.channel must be stable or rc')  # noqa: E701,E501
        if OCI_REPOSITORY_PATTERN.fullmatch(snapshot["target_repository"]) is None:
            raise ValueError(f'{location}.target_repository must use canonical OCI repository syntax')  # fmt: skip  # noqa: E501
        if OCI_TAG_PATTERN.fullmatch(snapshot["target_tag"]) is None:
            raise ValueError(f"{location}.target_tag must use strict OCI tag syntax")
        if (
            not isinstance(snapshot["index_digest"], str)
            or digest_pattern.fullmatch(snapshot["index_digest"]) is None
        ):
            raise ValueError(f"{location}.index_digest must be an exact sha256 digest")
        identity_version = snapshot['version'] if relaxed else str(_pep440_version(snapshot['version'], f'{location}.version'))  # fmt: skip  # noqa: E501
        identity = (snapshot['product_id'], snapshot['repository'], snapshot['tag'], identity_version, snapshot['channel'], snapshot['variant'])  # fmt: skip  # noqa: E501
        if identity in logical_identities: raise ValueError(f'{location} has duplicate logical upstream identity: {identity}')  # noqa: E701,E501
        logical_identities.add(identity)
        members = snapshot["members"]
        if not isinstance(members, dict) or not members: raise ValueError(f'{location}.members must be a non-empty object')  # noqa: E701,E501
        for architecture, member in members.items():
            member_location = f"{location}.members.{architecture}"
            if not isinstance(architecture, str) or not architecture:
                raise ValueError(f"{location}.members has an invalid architecture")
            if not isinstance(member, dict): raise ValueError(f'{member_location} must be an object')  # noqa: E701,E501
            missing = sorted(member_keys - set(member))
            extras = sorted(set(member) - member_keys)
            if missing or extras: raise ValueError(f'{member_location} requires exact key set; missing={missing}, extra={extras}')  # noqa: E701,E501
            for digest_name in sorted(member_keys):
                if (
                    not isinstance(member[digest_name], str)
                    or digest_pattern.fullmatch(member[digest_name]) is None
                ):
                    raise ValueError(f'{member_location}.{digest_name} must be an exact sha256 digest')  # fmt: skip  # noqa: E501


_LINKED_WHEEL_FIELDS = tuple('spec_id profile_id runner cpu_arch platform builder builder_sha256 build python_abi python_version wheel_version wheel_platform required_native forbidden_native allowed_dt_needed external_required_dependencies dependency_lock_sha256 dependency_lock runtime_requirements write_authority build_eligible'.split())  # fmt: skip  # noqa: E501


def _find_profile(
    catalog: dict[str, Any],
    product: dict[str, Any],
    snapshot: dict[str, Any],
    architecture: str,
    *,
    relaxed: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    variant = next((v for v in product['variants'] if v['id'] == snapshot['variant']), None)  # fmt: skip  # noqa: E501
    if variant is None:
        raise ValueError(f"snapshot variant {snapshot['variant']!r} is not declared by upstream product {product['id']!r}")  # fmt: skip  # noqa: E501
    version = None if relaxed else _pep440_version(snapshot["version"], "resolved upstream version")  # fmt: skip  # noqa: E501
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for rule in catalog["compatibility"]["rules"]:
        if product["id"] not in rule["upstream_products"] or snapshot["variant"] not in rule["variants"] or architecture not in rule["cpu_architectures"]:  # fmt: skip  # noqa: E501
            continue
        # PR/pinned path (relaxed) skips the catalog's version_specifier +
        # upstream-channel gates so an out-of-specifier / out-of-channel tag
        # can still match a profile by (product, variant, arch, accelerator).
        if not relaxed and (
            version not in _pep440_specifier(rule["version_specifier"], f"compatibility rule {rule['id']!r}")
            or snapshot["channel"] not in rule["upstream_channels"]
        ):
            continue
        for profile in catalog["wheel_profiles"]:
            if (
                architecture in profile["cpu_arch"]
                and profile["accelerator"] == rule["accelerator"]
                and profile["accelerator_runtime"] in rule["accelerator_runtimes"]
                and variant["npu_arch"] in profile["npu_arch"]
                and variant["npu_arch"] in rule["npu_architectures"]
                and any(item in rule["operating_systems"] for item in profile["os"])
                and profile["python_abi"] in rule["python_abis"]
            ):
                matches.append((profile, rule))
    if not matches:
        return None
    if len(matches) > 1:
        matched = sorted((f"{profile['id']} via {rule['id']}" for profile, rule in matches))  # fmt: skip  # noqa: E501
        raise ValueError(f"resolved upstream member matches overlapping wheel profiles: product={product['id']}, tag={snapshot['tag']}, architecture={architecture}, matches={matched}")  # fmt: skip  # noqa: E501
    return matches[0]


def _matching_profile(
    catalog: dict[str, Any],
    product: dict[str, Any],
    snapshot: dict[str, Any],
    architecture: str,
    *,
    relaxed: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    match = _find_profile(
        catalog, product, snapshot, architecture, relaxed=relaxed
    )
    if match is None:
        raise ValueError(f"resolved upstream member has no compatible wheel profile: product={product['id']}, tag={snapshot['tag']}, architecture={architecture}")  # fmt: skip  # noqa: E501
    return match


def candidate_exclusion_reason(
    catalog: dict[str, Any],
    product: dict[str, Any],
    candidate: dict[str, Any],
    *,
    relaxed: bool = False,
) -> str | None:
    product_variant = next(
        (item for item in product["variants"] if item["id"] == candidate["variant"]),
        None,
    )
    if product_variant is None:
        raise ValueError(
            f"snapshot variant {candidate['variant']!r} is not declared by upstream product {product['id']!r}"
        )
    profile_matches = [
        _find_profile(catalog, product, candidate, architecture, relaxed=relaxed)
        for architecture in product["required_cpu_architectures"]
    ]
    if any(match is None for match in profile_matches):
        return "compatibility-unsupported"
    return None


_PROFILE_DIRECT_FIELDS = tuple('accelerator accelerator_runtime python_version python_abi wheel_version wheel_platform binary_profile_id dist_name'.split())  # fmt: skip  # noqa: E501
_PROFILE_DEEPCOPY_FIELDS = tuple('validation_targets required_native forbidden_native allowed_dt_needed external_required_dependencies'.split())  # fmt: skip  # noqa: E501
_FAMILY_IDENTITY_KEYS = tuple('product_id repository tag variant index_digest target_repository target_tag'.split())  # fmt: skip  # noqa: E501
_RUNTIME_KEYS = tuple("repository tag version channel variant index_digest".split())


def release_topology(catalog: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    wheels = [
        {"profile_id": profile["id"], "cpu_arch": architecture}
        for profile in catalog["wheel_profiles"]
        for architecture in profile["cpu_arch"]
    ]
    families = [
        {"product_id": product["id"], "variant": variant["id"]}
        for product in catalog["upstream_products"]
        for variant in product["variants"]
    ]
    images = [
        {
            "product_id": product["id"],
            "variant": variant["id"],
            "cpu_arch": architecture,
        }
        for product in catalog["upstream_products"]
        for variant in product["variants"]
        for architecture in product["required_cpu_architectures"]
    ]
    return {
        "wheels": sorted(
            wheels, key=lambda item: (item["profile_id"], item["cpu_arch"])
        ),
        "families": sorted(
            families, key=lambda item: (item["product_id"], item["variant"])
        ),
        "images": sorted(
            images,
            key=lambda item: (
                item["product_id"],
                item["variant"],
                item["cpu_arch"],
            ),
        ),
    }


@dataclass(frozen=True, slots=True)
class ReleasePlan:

    lane: str
    wheel_tasks: list[dict[str, Any]]
    image_tasks: list[dict[str, Any]]
    family_tasks: list[dict[str, Any]]

    @classmethod
    def build(
        cls,
        catalog: dict[str, Any],
        resolved_upstreams: list[dict[str, Any]],
        *,
        lane: str,
        repository_root: Path = REPO_ROOT,
        relaxed: bool = False,
    ) -> "ReleasePlan":
        validate_catalog(catalog, repository_root=repository_root)
        _validate_resolved_builder_roots(catalog)
        validate_resolved_upstreams(resolved_upstreams, relaxed=relaxed)
        if lane not in catalog['lanes']: raise ValueError(f'unsupported validation lane: {lane}')  # noqa: E701,E501
        runtime_requirements = python_runtime_requirements(catalog)
        products = {item["id"]: item for item in catalog["upstream_products"]}
        write_authority = [] if lane == 'feature-candidate' else ['github-prerelease', 'ghcr-final-index', 'ghcr-private-staging']  # fmt: skip  # noqa: E501
        _version_sort = (lambda v: v) if relaxed else (lambda v: _pep440_version(v, 'resolved upstream version'))  # fmt: skip  # noqa: E501
        snapshots = sorted(resolved_upstreams, key=lambda item: (item['product_id'], _version_sort(item['version']), item['variant'], item['tag'], item['repository'], item['channel'], item['target_repository'], item['target_tag'], item['index_digest'], sha256_value(item['members'])))  # fmt: skip  # noqa: E501
        wheel_tasks_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        image_tasks: list[dict[str, Any]] = []
        family_tasks: list[dict[str, Any]] = []
        family_coordinates: set[str] = set()

        for snapshot in snapshots:
            product = products.get(snapshot["product_id"])
            if product is None: raise ValueError(f"resolved upstream references unknown product {snapshot['product_id']!r}")  # noqa: E701,E501
            if snapshot["repository"] != product["repository"]:
                raise ValueError('resolved upstream repository differs from catalog product')  # fmt: skip  # noqa: E501
            product_variant = next((item for item in product['variants'] if item['id'] == snapshot['variant']), None)  # fmt: skip  # noqa: E501
            if product_variant is None:
                raise ValueError(f"snapshot variant {snapshot['variant']!r} is not declared by upstream product {product['id']!r}")  # fmt: skip  # noqa: E501
            if not relaxed:
                version = _pep440_version(snapshot["version"], "resolved upstream version")
                if snapshot["channel"] == "stable" and version.is_prerelease:
                    raise ValueError("channel stable requires a final version")
                if snapshot["channel"] == "rc" and not (
                    version.pre
                    and version.pre[0] == "rc"
                    and not (version.epoch or version.dev or version.post or version.local)
                ):
                    raise ValueError("channel rc requires a plain rcN version")
                if version not in _pep440_specifier(
                    product["version_specifier"],
                    f"upstream product {product['id']!r}.version_specifier",
                ):
                    raise ValueError('resolved upstream version is outside product selection')  # fmt: skip  # noqa: E501
                if snapshot["channel"] not in product["channels"]:
                    raise ValueError("resolved upstream channel is not selected by product")
            members = snapshot["members"]
            missing = sorted(set(product["required_cpu_architectures"]) - set(members))
            if missing: raise ValueError(f"resolved upstream {snapshot['tag']} is missing required CPU architectures: {missing}")  # noqa: E701,E501
            coordinate = f"{snapshot['target_repository']}:{snapshot['target_tag']}"
            if coordinate in family_coordinates: raise ValueError(f'duplicate target image coordinate: {coordinate}')  # noqa: E701,E501
            family_coordinates.add(coordinate)
            family_identity = {k: snapshot[k] for k in _FAMILY_IDENTITY_KEYS}
            family_task_id = f"family-{sha256_value(family_identity).removeprefix('sha256:')}"  # fmt: skip  # noqa: E501
            family_images: list[dict[str, Any]] = []

            for architecture in sorted(members):
                profile, rule = _matching_profile(catalog, product, snapshot, architecture, relaxed=relaxed)  # fmt: skip  # noqa: E501
                wheel_key = (profile["id"], architecture)
                if wheel_key not in wheel_tasks_by_key:
                    declaration = {'spec_id': f"{profile['id']}-{architecture}", 'profile_id': profile['id'], **{k: profile[k] for k in _PROFILE_DIRECT_FIELDS}, 'npu_arch_or_na': profile['npu_arch'][0], 'os': profile['os'][0], 'cpu_arch': architecture, **{k: copy.deepcopy(profile[k]) for k in _PROFILE_DEEPCOPY_FIELDS}}  # fmt: skip  # noqa: E501
                    dependency_lock = {'build_tools': build_tool_dependency_records(catalog, profile['python_abi'], architecture), 'runtime_dependencies': runtime_dependency_records(catalog, profile['python_abi'], architecture)}  # fmt: skip  # noqa: E501
                    builder = profile["builders"][architecture]
                    wheel_identity = {'profile_id': profile['id'], 'cpu_arch': architecture, 'builder_sha256': sha256_value(builder), 'dependency_lock_sha256': sha256_value(dependency_lock)}  # fmt: skip  # noqa: E501
                    wheel_task: dict[str, Any] = {'task_id': f"wheel-{sha256_value(wheel_identity).removeprefix('sha256:')}", **declaration, 'declaration_sha256': sha256_value(declaration), 'runner': catalog['runner_map'][architecture], 'platform': f'linux/{architecture}', 'builder': builder, 'builder_sha256': sha256_value(builder), 'build': copy.deepcopy(profile['build']), 'dependency_lock_sha256': sha256_value(dependency_lock), 'dependency_lock': dependency_lock, 'runtime_requirements': copy.deepcopy(runtime_requirements), 'write_authority': write_authority, 'build_eligible': True}  # fmt: skip  # noqa: E501
                    wheel_task["artifact_name"] = f"ucm-wheel-{wheel_task['task_id']}"
                    wheel_task["task_sha256"] = sha256_value(wheel_task)
                    wheel_tasks_by_key[wheel_key] = wheel_task
                wheel_task = wheel_tasks_by_key[wheel_key]
                member = members[architecture]
                runtime = {'product_id': snapshot['product_id'], 'repository': snapshot['repository'], 'tag': snapshot['tag'], 'version': snapshot['version'], 'channel': snapshot['channel'], 'variant': snapshot['variant'], 'index_digest': snapshot['index_digest'], **member}  # fmt: skip  # noqa: E501
                image_identity = {'family_task_id': family_task_id, 'wheel_task_id': wheel_task['task_id'], 'cpu_arch': architecture, 'runtime_sha256': sha256_value(runtime)}  # fmt: skip  # noqa: E501
                image_task: dict[str, Any] = {'task_id': f"image-{sha256_value(image_identity).removeprefix('sha256:')}", 'family_task_id': family_task_id, 'wheel_task_id': wheel_task['task_id'], 'compatibility_rule_id': rule['id'], 'runtime': runtime, 'runtime_sha256': sha256_value(runtime), 'target_repository': snapshot['target_repository'], 'target_tag': snapshot['target_tag'], **{k: wheel_task[k] for k in _LINKED_WHEEL_FIELDS}}  # fmt: skip  # noqa: E501
                image_task["artifact_name"] = f"ucm-image-{image_task['task_id']}"
                image_task["wheel_artifact_name"] = wheel_task["artifact_name"]
                image_task["task_sha256"] = sha256_value(image_task)
                image_tasks.append(image_task)
                family_images.append(image_task)

            family_runtime = {k: snapshot[k] for k in _RUNTIME_KEYS}
            control_image = family_images[0]
            family_task: dict[str, Any] = {'task_id': family_task_id, 'product_id': snapshot['product_id'], 'control_task_id': control_image['task_id'], 'control_arch': control_image['cpu_arch'], 'control_runner': control_image['runner'], 'runner': [item['runner'] for item in family_images], 'cpu_arch': [item['cpu_arch'] for item in family_images], 'platform': [item['platform'] for item in family_images], 'builder': [item['builder'] for item in family_images], 'builder_sha256': [item['builder_sha256'] for item in family_images], 'runtime': family_runtime, 'runtime_sha256': sha256_value(family_runtime), 'snapshot_sha256': sha256_value(snapshot), 'target_repository': snapshot['target_repository'], 'target_tag': snapshot['target_tag'], 'image_task_ids': [item['task_id'] for item in family_images], 'wheel_task_ids': {item['cpu_arch']: item['wheel_task_id'] for item in family_images}, 'member_set_sha256': sha256_value([item['task_sha256'] for item in family_images]), 'write_authority': write_authority}  # fmt: skip  # noqa: E501
            family_task["artifact_name"] = f"ucm-family-{family_task['task_id']}"
            family_task["task_sha256"] = sha256_value(family_task)
            family_tasks.append(family_task)

        wheel_tasks = sorted(wheel_tasks_by_key.values(), key=lambda item: (item['profile_id'], item['cpu_arch']))  # fmt: skip  # noqa: E501
        limits = catalog["matrix_limits"]
        cardinalities = {'wheel_tasks': len(wheel_tasks), 'image_tasks': len(image_tasks), 'family_tasks': len(family_tasks)}  # fmt: skip  # noqa: E501
        for task_kind, count in cardinalities.items():
            limit = limits[f"max_{task_kind}"]
            if count > limit: raise ValueError(f'matrix limit max_{task_kind}={limit} exceeded by exact generated set of {count}')  # noqa: E701,E501
        return cls(lane=lane, wheel_tasks=wheel_tasks, image_tasks=image_tasks, family_tasks=family_tasks)  # fmt: skip  # noqa: E501


def expand_release_plan(
    catalog: dict[str, Any],
    resolved_upstreams: list[dict[str, Any]],
    *,
    lane: str,
    repository_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    plan = ReleasePlan.build(catalog, resolved_upstreams, lane=lane, repository_root=repository_root)  # fmt: skip  # noqa: E501
    result: dict[str, Any] = {'schema_version': 2, 'kind': 'ucm-resolved-build-plan', 'lane': plan.lane, 'wheel_tasks': plan.wheel_tasks, 'image_tasks': plan.image_tasks, 'family_tasks': plan.family_tasks, 'cardinalities': {'wheel_tasks': len(plan.wheel_tasks), 'image_tasks': len(plan.image_tasks), 'family_tasks': len(plan.family_tasks)}}  # fmt: skip  # noqa: E501
    result["plan_sha256"] = sha256_value(result)
    return result


RELEASE_KEYS = frozenset('kind schema_version image_revision source lanes runner_map upstream_products chart publish'.split())  # fmt: skip  # noqa: E501
OPTIONAL_CATALOG_KEYS = frozenset('ucm_version matrix_limits scan_limits python_runtime_dependencies python_build_lock wheel_build_requirements wheel_runtime_requirements builder_checks backend_contracts'.split())  # fmt: skip  # noqa: E501
LANES = ("feature-candidate", "protected-tag")


def _exact_keys(
    value: dict[str, Any],
    expected: set[str],
    location: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(expected - set(value))
    extras = sorted(set(value) - expected - optional)
    if missing or extras: raise ValueError(f'{location} requires exact key set; missing={missing}, extra={extras}')  # noqa: E701,E501


def _validate_cross_config(
    release: dict[str, Any], *, repository_root: Path = REPO_ROOT
) -> None:
    validate_catalog(release, repository_root=repository_root)
    publish = compute_publish_plan(release)
    if any(publish[channel]["enabled"] for channel in PUBLISH_CHANNELS[:-1]) and not publish["github_release"]["enabled"]:
        raise ValueError("enabled public channels require the GitHub Release Draft barrier")
    if publish["dockerhub"]["enabled"] and not publish["ghcr"]["enabled"]:
        raise ValueError("Docker Hub publication requires GHCR source publication")
    if "wheel_profiles" not in release:
        products = {product["id"] for product in release["upstream_products"]}
        selectors: set[tuple[str, str]] = set()
        for case in release["chart"]["validation_cases"]:
            selector = (case["product_id"], case["variant"])
            if case["product_id"] not in products:
                raise ValueError(
                    "Chart validation references an unknown upstream product"
                )
            if selector in selectors:
                raise ValueError(
                    "Chart validation product/variant selectors must be unique"
                )
            selectors.add(selector)
        return
    profiles = release["wheel_profiles"]
    for profile in profiles:
        architectures = set(profile["cpu_arch"])
        if architectures != set(profile["builders"]):
            raise ValueError(f"wheel profile {profile['id']!r} builder architectures do not match cpu_arch")  # fmt: skip  # noqa: E501
        missing_runners = sorted(architectures - set(release["runner_map"]))
        if missing_runners: raise ValueError(f"wheel profile {profile['id']!r} has no runner for {missing_runners}")  # noqa: E701,E501
    products = {product["id"]: product for product in release["upstream_products"]}
    chart_selectors: set[tuple[str, str]] = set()
    for case in release["chart"]["validation_cases"]:
        selector = (case["product_id"], case["variant"])
        product = products.get(case["product_id"])
        if product is None or case["variant"] not in {
            variant["id"] for variant in product["variants"]
        }:
            raise ValueError('Chart validation must select a declared upstream product variant')  # fmt: skip  # noqa: E501
        if selector in chart_selectors: raise ValueError('Chart validation product/variant selectors must be unique')  # noqa: E701,E501
        chart_selectors.add(selector)


def load_catalog(
    release_path: Path = DEFAULT_RELEASE,
    schema_dir: Path = DEFAULT_SCHEMA_DIR,
    *,
    repository_root: Path = REPO_ROOT,
    repository: str | None = None,
    version_override: str | None = None,
) -> dict[str, Any]:
    config_schema = load_json(schema_dir / "config.schema.json")
    release = load_yaml(release_path)
    resolved_repository = resolve_repository(repository, repository_root=repository_root)  # fmt: skip  # noqa: E501
    is_formal_policy = (
        release.get("kind") == "ucm-release-policy"
        and release.get("schema_version") == 5
    )
    if is_formal_policy:
        from . import policy as release_policy

        formal_policy = release_policy.load(
            release_path, schema_path=schema_dir / "config.schema.json"
        )
        chart_source = formal_policy["release"]["chart"]["source"]
        chart_document = load_yaml(repository_root / chart_source / "Chart.yaml")
        chart_name = chart_document.get("name")
        if not isinstance(chart_name, str) or not chart_name:
            raise ValueError("Chart.yaml name must be a non-empty string")
        release = release_policy.compatibility_projection(
            formal_policy,
            chart_name=chart_name,
            repository=resolved_repository,
        )
    release = resolve_owner_templates(release, repository=resolved_repository)
    validate_schema(release, config_schema)
    missing = sorted(RELEASE_KEYS - set(release))
    extras = sorted(set(release) - RELEASE_KEYS - OPTIONAL_CATALOG_KEYS)
    if missing or extras: raise ValueError(f'release.yaml requires exact key set; missing={missing}, extra={extras}')  # noqa: E701,E501
    release["source"]["repository"] = resolved_repository
    version = version_override or read_version(repository_root / "version.ini")
    release["ucm_version"] = version
    release["source"]["release_tag"] = f"v{version}"
    release["chart"]["version"] = derive_chart_version(version)
    release["chart"]["app_version"] = version
    image_suffix = f"-ucm-{_oci_tag_version(version)}"
    if not is_formal_policy:
        image_suffix += f"-r{release.get('image_revision', 1)}"
    for product in release.get("upstream_products", []):
        product["target_tag_suffix"] = image_suffix
    for profile in release.get("wheel_profiles", []):

        profile["wheel_version"] = version  # fmt: skip  # noqa: E501
        profile.setdefault("dist_name", "uc-manager")  # fmt: skip  # noqa: E501
    chart = load_yaml(repository_root / release["chart"]["source"] / "Chart.yaml")
    if chart.get('name') != release['chart']['name']: raise ValueError('Chart name does not match release.yaml')  # noqa: E701,E501
    _validate_cross_config(release, repository_root=repository_root)
    return release


PUBLISH_CHANNELS = ("pypi", "ghcr", "dockerhub", "chart_oci", "github_release")
_PUBLISH_DECISION_KEYS = frozenset({"requested", "enabled", "disposition"})
PUBLISH_CHANNEL_KEYS = {
    "pypi": _PUBLISH_DECISION_KEYS | {"index"},
    "ghcr": _PUBLISH_DECISION_KEYS | {"namespace"},
    "dockerhub": _PUBLISH_DECISION_KEYS | {"namespace"},
    "chart_oci": _PUBLISH_DECISION_KEYS | {"namespace"},
    "github_release": _PUBLISH_DECISION_KEYS,
}


def compute_publish_plan(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    publish = catalog.get("publish")
    if not isinstance(publish, dict) or set(publish) != set(PUBLISH_CHANNELS):
        raise ValueError("publish configuration requires the exact five channels")
    for channel in PUBLISH_CHANNELS:
        config = publish[channel]
        if not isinstance(config, dict) or set(config) != PUBLISH_CHANNEL_KEYS[channel]:
            raise ValueError(f"publish channel {channel} configuration is malformed")
        if not isinstance(config["requested"], bool) or not isinstance(
            config["enabled"], bool
        ):
            raise ValueError(
                f"publish channel {channel} requested/enabled must be boolean"
            )
        disposition = config["disposition"]
        expected_state = {
            "publish": (True, True),
            "scope-skipped": (True, False),
            "disabled": (False, False),
        }.get(disposition)
        if expected_state != (config["requested"], config["enabled"]):
            raise ValueError(
                f"publish channel {channel} has an invalid publication decision"
            )
        for field in PUBLISH_CHANNEL_KEYS[channel] - _PUBLISH_DECISION_KEYS:
            if not isinstance(config[field], str) or not config[field]:
                raise ValueError(f"publish channel {channel} {field} must be non-empty")
    return {channel: copy.deepcopy(publish[channel]) for channel in PUBLISH_CHANNELS}


def _git_output(repository_root: Path, *arguments: str) -> str | None:
    completed = subprocess.run(['git', '-C', str(repository_root), *arguments], text=True, capture_output=True, check=False)  # fmt: skip  # noqa: E501
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_commit(repository_root: Path, revision: str) -> str | None:
    commit = _git_output(repository_root, 'rev-parse', '--verify', f'{revision}^{{commit}}')  # fmt: skip  # noqa: E501
    if commit is None or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        return None
    return commit


def _origin_repository(remote_url: str | None) -> str | None:
    if remote_url is None:
        return None
    prefixes = ("https://github.com/", "git@github.com:")
    for prefix in prefixes:
        if remote_url.startswith(prefix):
            repository = remote_url.removeprefix(prefix).removesuffix(".git")
            if re.fullmatch(r"[^/]+/[^/]+", repository):
                return repository
    return None


_OWNER_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
_KNOWN_OWNER_PLACEHOLDERS = frozenset({"owner", "repo"})
_REPOSITORY_IDENTITY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def resolve_repository(
    repository: str | None = None,
    *,
    repository_root: Path = REPO_ROOT,
) -> str:
    if repository:
        candidate = repository.strip()
    else:
        candidate = os.environ.get("GITHUB_REPOSITORY", "").strip()
        if not candidate:
            remote = _git_output(repository_root, "remote", "get-url", "origin")
            candidate = _origin_repository(remote).strip() if remote else ""
    if not candidate or _REPOSITORY_IDENTITY_RE.fullmatch(candidate) is None:
        raise ValueError('could not resolve the running repository; pass --repository or set GITHUB_REPOSITORY (local dev infers from the origin remote)')  # fmt: skip  # noqa: E501
    return candidate


def resolve_owner_templates(catalog: Any, *, repository: str) -> Any:
    parts = repository.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"repository must be 'owner/name', got: {repository!r}")
    owner = parts[0].lower()
    repo = parts[1].lower()
    return _walk_owner_templates(catalog, owner=owner, repo=repo)


def _walk_owner_templates(value: Any, *, owner: str, repo: str) -> Any:
    if isinstance(value, str):
        return _substitute_owner(value, owner=owner, repo=repo)
    if isinstance(value, dict):
        return {key: _walk_owner_templates(item, owner=owner, repo=repo) for key, item in value.items()}  # fmt: skip  # noqa: E501
    if isinstance(value, list):
        return [_walk_owner_templates(item, owner=owner, repo=repo) for item in value]
    return value


def _substitute_owner(text: str, *, owner: str, repo: str) -> str:
    if "{" not in text:
        return text
    placeholders = _OWNER_PLACEHOLDER.findall(text)
    if not placeholders:
        return text
    unknown = sorted({name for name in placeholders if name not in _KNOWN_OWNER_PLACEHOLDERS})  # fmt: skip  # noqa: E501
    if unknown:
        raise ValueError(f'unknown owner template placeholder(s) {unknown} in {text!r}; recognised: {sorted(_KNOWN_OWNER_PLACEHOLDERS)}')  # fmt: skip  # noqa: E501
    return text.replace("{owner}", owner).replace("{repo}", repo)


def _is_ancestor(repository_root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(['git', '-C', str(repository_root), 'merge-base', '--is-ancestor', ancestor, descendant], text=True, capture_output=True, check=False)  # fmt: skip  # noqa: E501
    return completed.returncode == 0


TAG_PREFLIGHT_AUTHORITY_FIELDS = frozenset('repository staging_repository default_branch release_tag ucm_version'.split())  # fmt: skip  # noqa: E501


def _tag_preflight_live(
    *,
    lane: str,
    authority: dict[str, Any],
    repository_root: Path | None = None,
) -> dict[str, Any]:
    if repository_root is None: repository_root = REPO_ROOT  # noqa: E701
    if lane not in LANES: raise ValueError(f'unsupported validation lane: {lane}')  # noqa: E701,E501
    if (
        not isinstance(authority, dict)
        or set(authority)
        not in {
            frozenset(TAG_PREFLIGHT_AUTHORITY_FIELDS),
            frozenset(TAG_PREFLIGHT_AUTHORITY_FIELDS | {"commit"}),
        }
        or any(
            not isinstance(authority[field], str) or not authority[field]
            for field in TAG_PREFLIGHT_AUTHORITY_FIELDS
        )
    ):
        raise ValueError("release preflight authority is malformed")
    repository_owner = authority["repository"].split("/", 1)[0]
    if lane == "feature-candidate":
        checks = {"feature_zero_write": True}
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed: raise ValueError(f'release preflight failed: {failed}')  # noqa: E701
        result: dict[str, Any] = {'schema_version': 1, 'kind': 'ucm-tag-preflight', 'lane': lane, 'repository': authority['repository'], 'repository_owner': repository_owner, 'ref': None, 'ref_type': None, 'ref_name': None, 'source_sha': None, 'default_branch': authority['default_branch'], 'checks': checks, 'publication_allowed': False, 'write_authority': []}  # fmt: skip  # noqa: E501
        result["preflight_sha256"] = sha256_value(result)
        return result

    context_names = 'GITHUB_ACTIONS GITHUB_ACTOR GITHUB_EVENT_NAME GITHUB_EVENT_PATH GITHUB_REF GITHUB_REF_NAME GITHUB_REF_PROTECTED GITHUB_REF_TYPE GITHUB_REPOSITORY GITHUB_REPOSITORY_OWNER GITHUB_SHA GITHUB_TRIGGERING_ACTOR'.split()  # fmt: skip  # noqa: E501
    context = {name: os.environ.get(name, "") for name in context_names}
    event_path = Path(context["GITHUB_EVENT_PATH"])
    if not context["GITHUB_EVENT_PATH"] or not event_path.is_file():
        raise ValueError("release preflight failed: ['github_event_path']")
    event = load_json(event_path)
    event_repository = event.get("repository")
    if not isinstance(event_repository, dict): event_repository = {}  # noqa: E701
    event_owner = event_repository.get("owner")
    if not isinstance(event_owner, dict): event_owner = {}  # noqa: E701
    event_sender = event.get("sender")
    if not isinstance(event_sender, dict): event_sender = {}  # noqa: E701

    release_tag = authority["release_tag"]
    tag_ref = f"refs/tags/{release_tag}"
    default_branch_ref = f"refs/remotes/origin/{authority['default_branch']}"
    source_sha = context["GITHUB_SHA"]
    checked_head_sha = _git_commit(repository_root, "HEAD")
    tag_commit_sha = _git_commit(repository_root, tag_ref)
    default_branch_sha = _git_commit(repository_root, default_branch_ref)
    source_commit_sha = _git_commit(repository_root, source_sha) if re.fullmatch('[0-9a-f]{40}', source_sha) else None  # fmt: skip  # noqa: E501
    worktree_root = _git_output(repository_root, "rev-parse", "--show-toplevel")
    origin_repository = _origin_repository(_git_output(repository_root, 'remote', 'get-url', 'origin'))  # fmt: skip  # noqa: E501
    event_after = event.get('after')
    event_after_commit = _git_commit(repository_root, event_after) if event_after else None  # fmt: skip  # noqa: E501
    checks = {'actor': context['GITHUB_ACTOR'] == repository_owner, 'checked_head': checked_head_sha == source_sha, 'default_branch': event_repository.get('default_branch') == authority['default_branch'], 'default_branch_ancestry': True, 'event_actor': event_sender.get('login') == context['GITHUB_ACTOR'], 'event_name': context['GITHUB_EVENT_NAME'] == 'push', 'event_owner': event_owner.get('login') == context['GITHUB_REPOSITORY_OWNER'], 'event_ref': event.get('ref') == context['GITHUB_REF'], 'event_repository': event_repository.get('full_name') == context['GITHUB_REPOSITORY'], 'event_source_sha': event_after is None or event_after_commit == source_sha, 'github_actions': context['GITHUB_ACTIONS'] == 'true', 'origin_repository': origin_repository == authority['repository'], 'owner': context['GITHUB_REPOSITORY_OWNER'] == repository_owner, 'ref': context['GITHUB_REF'] == tag_ref, 'ref_name': context['GITHUB_REF_NAME'] == release_tag, 'ref_protected': True, 'ref_type': context['GITHUB_REF_TYPE'] == 'tag', 'repository': context['GITHUB_REPOSITORY'] == authority['repository'], 'repository_root': worktree_root is not None and Path(worktree_root).resolve() == repository_root.resolve(), 'source_sha': source_commit_sha == source_sha, 'frozen_source': authority.get('commit', source_sha) == source_sha, 'tag_commit': tag_commit_sha == source_sha}  # fmt: skip  # noqa: E501
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed: raise ValueError(f'release preflight failed: {failed}')  # noqa: E701
    result = {'schema_version': 1, 'kind': 'ucm-tag-preflight', 'lane': lane, 'repository': context['GITHUB_REPOSITORY'], 'repository_owner': context['GITHUB_REPOSITORY_OWNER'], 'actor': context['GITHUB_ACTOR'], 'triggering_actor': context['GITHUB_TRIGGERING_ACTOR'], 'ref': context['GITHUB_REF'], 'ref_type': context['GITHUB_REF_TYPE'], 'ref_name': context['GITHUB_REF_NAME'], 'source_sha': source_sha, 'tag_commit_sha': tag_commit_sha, 'checked_head_sha': checked_head_sha, 'default_branch': authority['default_branch'], 'default_branch_ref': default_branch_ref, 'default_branch_sha': default_branch_sha, 'event_payload_sha256': sha256_value(event), 'checks': checks, 'publication_allowed': True, 'write_authority': ['github-prerelease', 'ghcr-final-index', 'ghcr-private-staging']}  # fmt: skip  # noqa: E501
    result["preflight_sha256"] = sha256_value(result)
    return result


def tag_preflight(
    *,
    lane: str,
    release_path: Path = DEFAULT_RELEASE,
    schema_dir: Path = DEFAULT_SCHEMA_DIR,
    authority: dict[str, Any] | None = None,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    if authority is None:
        release = load_catalog(release_path, schema_dir)
        authority = {field: release['source'][field] for field in ('repository', 'staging_repository', 'default_branch', 'release_tag')}  # fmt: skip  # noqa: E501
        authority = {**authority, 'ucm_version': release['ucm_version']}  # fmt: skip  # noqa: E501
        # catalog-planner mode: return resolved authority without live git/event checks
        result = {'schema_version': 1, 'kind': 'ucm-tag-preflight', 'lane': lane, 'authority': authority, 'publication_allowed': False, 'write_authority': []}  # fmt: skip  # noqa: E501
        result['preflight_sha256'] = sha256_value(result)
        return result
    if repository_root is None: repository_root = REPO_ROOT  # noqa: E701
    return _tag_preflight_live(lane=lane, authority=authority, repository_root=repository_root)  # fmt: skip  # noqa: E501
# fmt: on
