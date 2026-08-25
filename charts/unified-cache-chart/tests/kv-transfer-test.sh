#!/bin/bash
# KV Transfer 多 connector 的 Helm 渲染/失败矩阵。
# 依赖：helm、python3(+PyYAML)、bash。用法：bash tests/kv-transfer-test.sh
#
# 覆盖：
#   - MooncakeConnectorV1 / MooncakeHybridConnector 的 UCM on/off；
#   - NixlConnector（UCM off）以及 NIXL+UCM 拒绝；
#   - connector 与 routerType 分离、严格大小写和旧字段迁移门禁；
#   - 动态 identity 模板、MultiConnector 字段位置、Downward API 标签；
#   - --kv-transfer-config 只由共享 entrypoint 追加一次。
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH

CHART_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$CHART_DIR"

# 所有 case 都从同一个最小 1P1D、entry+worker fixture 派生，避免测试依赖
# models/ 下某个具体模型、镜像或存储后端。UCM off case 仍保留非空 config，
# 用于验证 enabled=false 确实是唯一有效开关。
python3 - "$WORK" <<'PYEOF'
import copy
import json
import os
import sys

import yaml


work = sys.argv[1]


def make_case(connector="MooncakeConnectorV1", router_type="mooncake", ucm=False):
    identity = {"engineIdBase": 0}
    if connector in ("MooncakeConnectorV1", "MooncakeHybridConnector"):
        identity.update({"kvPortBase": 36000, "instanceStride": 100})

    prefill_args = """\
--tensor_parallel'-size' 1
--data-parallel-size 2
--pipeline-"parallel-size" 2
--data-parallel-size-local 1
--max-model-len 1024
"""
    decode_args = prefill_args.replace("--tensor_parallel'-size' 1", "--tensor-parallel-size 1")
    decode_args = decode_args.replace('--pipeline-"parallel-size" 2', "--pipeline-parallel-size 1")
    return {
        "servingEngineSpec": {
            "enableEngine": True,
            "hostNetwork": False,
            "modelSpec": {
                "name": "kv-test",
                "modelPath": "/models/kv-test",
                "modelName": "kv-test",
                "pd": {
                    "prefill": "prefill",
                    "decode": "decode",
                    "antiAffinity": True,
                    "kvTransfer": {
                        "connector": connector,
                        "routerType": router_type,
                        "identity": identity,
                    },
                },
                "router": {
                    "enabled": True,
                    "inferenceEngine": "vLLM",
                },
                "roles": [
                    {
                        "name": "prefill",
                        "replicas": 1,
                        "workerReplicas": 1,
                        "resources": {
                            "limits": {"cpu": "1", "memory": "1Gi"},
                            "requests": {"cpu": "1", "memory": "1Gi"},
                        },
                        "vllmArgs": prefill_args,
                    },
                    {
                        "name": "decode",
                        "replicas": 1,
                        "workerReplicas": 1,
                        "resources": {
                            "limits": {"cpu": "1", "memory": "1Gi"},
                            "requests": {"cpu": "1", "memory": "1Gi"},
                        },
                        "vllmArgs": decode_args,
                    },
                ],
                "unifiedcacheConfig": {
                    "enabled": bool(ucm),
                    "kvcsStoreIdAutoDetect": False,
                    "config": {
                        "ucm_connectors": [
                            {
                                "ucm_connector_name": "UcmPipelineStore",
                                "ucm_connector_config": {
                                    "store_pipeline": "Cache|Posix",
                                    "cache_buffer_capacity_gb": 1,
                                },
                            }
                        ]
                    },
                },
                "storage": {
                    "unifiedcacheStorage": [
                        {
                            "name": "ucm-cache",
                            "mountPath": "/mnt/ucm-cache",
                            "hostPath": {
                                "path": "/tmp/uc-stack-kv-test",
                                "type": "DirectoryOrCreate",
                            },
                        }
                    ],
                    "extraStorage": [],
                },
            },
        }
    }


cases = {
    "v1-off": make_case("MooncakeConnectorV1", "mooncake", False),
    "v1-on": make_case("MooncakeConnectorV1", "mooncake", True),
    "hybrid-off": make_case("MooncakeHybridConnector", "mooncake", False),
    "hybrid-on": make_case("MooncakeHybridConnector", "mooncake", True),
    "nixl-off": make_case("NixlConnector", "nixl", False),
    "nixl-on": make_case("NixlConnector", "nixl", True),
    "router-mismatch": make_case("MooncakeConnectorV1", "nixl", False),
    "unknown-connector": make_case("UnknownConnector", "mooncake", False),
    "connector-case-error": make_case("mooncakeconnectorv1", "mooncake", False),
}

default_ucm = make_case("MooncakeConnectorV1", "mooncake", True)
default_ucm["servingEngineSpec"]["modelSpec"]["unifiedcacheConfig"].pop("enabled")
cases["v1-default-on"] = default_ucm

alias_ucm = make_case("MooncakeConnectorV1", "mooncake", True)
alias_ucm["servingEngineSpec"]["modelSpec"]["unifiedcacheConfig"].pop("enabled")
alias_ucm["servingEngineSpec"]["modelSpec"]["unifiedcacheConfig"]["enable"] = True
cases["v1-alias-on"] = alias_ucm

missing_connector = make_case()
missing_connector["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"].pop("connector")
cases["missing-connector"] = missing_connector

missing_router = make_case()
missing_router["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"].pop("routerType")
cases["missing-router"] = missing_router

for managed_connector in ("MultiConnector", "UCMConnector"):
    cases[f"managed-{managed_connector}"] = make_case(
        managed_connector, "mooncake", False
    )

missing_engine = make_case()
missing_engine["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"]["identity"].pop("engineIdBase")
cases["missing-engine"] = missing_engine

unknown_identity = make_case()
unknown_identity["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"]["identity"]["engine_id"] = 0
cases["unknown-identity"] = unknown_identity

negative_engine = make_case()
negative_engine["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"]["identity"]["engineIdBase"] = -1
cases["negative-engine"] = negative_engine

fractional_engine = make_case()
fractional_engine["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"]["identity"]["engineIdBase"] = 0.5
cases["fractional-engine"] = fractional_engine

string_port = make_case()
string_port["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"]["identity"]["kvPortBase"] = "36000"
cases["string-port"] = string_port

fractional_port = make_case()
fractional_port["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"]["identity"]["kvPortBase"] = 36000.5
cases["fractional-port"] = fractional_port

missing_port = make_case()
missing_port["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"]["identity"].pop("kvPortBase")
cases["missing-port"] = missing_port

fractional_stride = make_case()
fractional_stride["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"]["identity"]["instanceStride"] = 100.5
cases["fractional-stride"] = fractional_stride

small_stride = make_case()
small_stride["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"]["identity"]["instanceStride"] = 99
cases["small-stride"] = small_stride

parallel_stride = make_case()
parallel_stride["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"]["identity"]["instanceStride"] = 128
for role in parallel_stride["servingEngineSpec"]["modelSpec"]["roles"]:
    pp_size = 2 if role["name"] == "prefill" else 1
    role["vllmArgs"] = (
        "--tensor_parallel'-size' 8\n"
        "--data-parallel-size 4\n"
        f"--pipeline-parallel-size {pp_size}\n"
        "--prefill-context-parallel-size 2\n"
        "--decode-context-parallel-size 2\n"
    )
cases["parallel-stride"] = parallel_stride

decode_pp = make_case()
decode_pp["servingEngineSpec"]["modelSpec"]["roles"][1]["vllmArgs"] = (
    "--tensor-parallel-size 1\n"
    "--data-parallel-size 2\n"
    "--pipeline-parallel-size 2\n"
)
cases["decode-pp"] = decode_pp

managed_arg_inline = make_case()
managed_arg_inline["servingEngineSpec"]["modelSpec"]["roles"][0]["vllmArgs"] += (
    "--tensor-parallel-size 1 --kv-transfer-config '{}'\n"
)
cases["managed-arg-inline"] = managed_arg_inline

managed_arg_quoted = make_case()
managed_arg_quoted["servingEngineSpec"]["modelSpec"]["roles"][0]["vllmArgs"] += (
    "--kv-transfer'-config' '{}'\n"
)
cases["managed-arg-quoted"] = managed_arg_quoted

managed_arg_underscore = make_case()
managed_arg_underscore["servingEngineSpec"]["modelSpec"]["roles"][0]["vllmArgs"] += (
    "--kv_transfer_config '{}'\n"
)
cases["managed-arg-underscore"] = managed_arg_underscore

managed_arg_abbrev = make_case()
managed_arg_abbrev["servingEngineSpec"]["modelSpec"]["roles"][0]["vllmArgs"] += (
    "--kv-transfer-conf '{}'\n"
)
cases["managed-arg-abbrev"] = managed_arg_abbrev

managed_vllm_config = make_case()
managed_vllm_config["servingEngineSpec"]["modelSpec"]["roles"][0]["vllmArgs"] += (
    "--config /models/opaque-vllm.yaml\n"
)
cases["managed-vllm-config"] = managed_vllm_config

managed_arg_dotted = make_case()
managed_arg_dotted["servingEngineSpec"]["modelSpec"]["roles"][0]["vllmArgs"] += (
    "--kv_transfer_config.kv_connector MooncakeConnectorV1\n"
)
cases["managed-arg-dotted"] = managed_arg_dotted

for alias, spelling in {
    "dpa": "-dpa attacker",
    "dpp": "-dpp=9999",
    "dpr": "-dpr 9",
}.items():
    managed_short = make_case()
    managed_short["servingEngineSpec"]["modelSpec"]["roles"][0]["vllmArgs"] += (
        spelling + "\n"
    )
    cases[f"managed-short-{alias}"] = managed_short

layout_abbrev = make_case()
layout_abbrev["servingEngineSpec"]["modelSpec"]["roles"][0]["vllmArgs"] = (
    "--tensor-parallel-s 128\n"
)
cases["layout-abbrev"] = layout_abbrev

kv_offloading_size = make_case()
kv_offloading_size["servingEngineSpec"]["modelSpec"]["roles"][0]["vllmArgs"] += (
    "--kv-offloading-size 1\n"
)
cases["kv-offloading-size"] = kv_offloading_size

kv_offloading_backend = make_case()
kv_offloading_backend["servingEngineSpec"]["modelSpec"]["roles"][0]["vllmArgs"] += (
    "--kv_offloading_back cpu\n"
)
cases["kv-offloading-backend"] = kv_offloading_backend

native_offloading = make_case()
native_model = native_offloading["servingEngineSpec"]["modelSpec"]
native_model.pop("pd")
native_model["router"]["enabled"] = False
native_model["unifiedcacheConfig"]["enabled"] = False
native_role = native_model["roles"][0]
native_role["name"] = "engine"
native_role["vllmArgs"] += "--kv-offloading-size 1\n"
native_model["roles"] = [native_role]
cases["native-offloading-no-chart-kv"] = native_offloading

dp_env_sanitized = make_case("MooncakeHybridConnector", "mooncake", False)
dp_env_model = dp_env_sanitized["servingEngineSpec"]["modelSpec"]
dp_env_model["env"] = [{"name": "VLLM_DP_SIZE", "value": "128"}]
dp_env_sanitized["nodeTopologyConfig"] = {
    "node-a": {"VLLM_DP_RANK": "17", "VLLM_DP_MASTER_PORT": "9999"}
}
cases["kv-dp-env-sanitized"] = dp_env_sanitized

for managed_env in (
    "UC_PD_GROUP_NAME",
    "UC_PD_ROLE_ID",
    "UC_USES_UCM",
    "UC_SKIP_KV_CONNECTOR_REGISTRY_PROBE",
    "VLLM_ARGS_FILE",
):
    case = make_case()
    case["servingEngineSpec"]["modelSpec"]["env"] = [
        {"name": managed_env, "value": "override"}
    ]
    cases[f"managed-env-{managed_env.lower().replace('_', '-')}"] = case

    topology_case = make_case()
    topology_case["nodeTopologyConfig"] = {
        "node-a": {managed_env: "override"}
    }
    cases[
        f"managed-topology-env-{managed_env.lower().replace('_', '-')}"
    ] = topology_case

topology_shell_key = make_case()
topology_shell_key["nodeTopologyConfig"] = {
    "node-a": {"export UC_PD_GROUP_NAME": "forged-0"}
}
cases["topology-shell-key"] = topology_shell_key

overflow_port = make_case()
overflow_port["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"]["identity"]["kvPortBase"] = 65500
cases["overflow-port"] = overflow_port

nixl_ports = make_case("NixlConnector", "nixl", False)
nixl_ports["servingEngineSpec"]["modelSpec"]["pd"]["kvTransfer"]["identity"].update(
    {"kvPortBase": 36000, "instanceStride": 100}
)
cases["nixl-port-fields"] = nixl_ports

same_roles = make_case()
same_roles["servingEngineSpec"]["modelSpec"]["pd"]["decode"] = "prefill"
cases["same-roles"] = same_roles

unknown_role = make_case()
unknown_role["servingEngineSpec"]["modelSpec"]["pd"]["decode"] = "missing"
cases["unknown-role"] = unknown_role

empty_ucm = make_case("MooncakeConnectorV1", "mooncake", True)
empty_ucm["servingEngineSpec"]["modelSpec"]["unifiedcacheConfig"]["config"] = {}
cases["empty-ucm"] = empty_ucm

for switch_mode in ("default", "enabled", "alias"):
    invalid_ucm = make_case("MooncakeHybridConnector", "mooncake", True)
    ucm = invalid_ucm["servingEngineSpec"]["modelSpec"]["unifiedcacheConfig"]
    ucm["config"] = "garbage"
    if switch_mode == "default":
        ucm.pop("enabled")
    elif switch_mode == "alias":
        ucm.pop("enabled")
        ucm["enable"] = True
    cases[f"invalid-ucm-config-{switch_mode}"] = invalid_ucm

invalid_ucm_switch = make_case()
invalid_ucm_switch["servingEngineSpec"]["modelSpec"]["unifiedcacheConfig"]["enabled"] = "false"
cases["invalid-ucm-switch"] = invalid_ucm_switch

# 旧字段必须按“字段存在”判断。每个字段同时覆盖 false 和空值，防止用
# truthiness 绕过迁移门禁。
for key, values in {
    "connector": [False, ""],
    "mooncakePort": [False, {}],
    "ucm": [False, {}],
}.items():
    for index, value in enumerate(values):
        case = make_case("MooncakeConnectorV1", "mooncake", False)
        case["servingEngineSpec"]["modelSpec"]["pd"][key] = value
        cases[f"legacy-{key}-{index}"] = case

master_conflict = make_case("NixlConnector", "nixl", False)
master_conflict["servingEngineSpec"]["modelSpec"]["pd"]["mooncake"] = {
    "master": {"enabled": True}
}
# 把顶层 master 配成完整可用，确保失败原因只能是 NIXL/routerType 冲突。
master_conflict["mooncakeMaster"] = {
    "enabled": True,
    "create": True,
    "client": {
        "config": json.dumps(
            {
                "metadata_server": "P2PHANDSHAKE",
                "global_segment_size": "1GB",
                "local_buffer_size": "128MB",
                "protocol": "rdma",
                "device_name": "",
            }
        )
    },
}
cases["nixl-mooncake-master"] = master_conflict

disabled_engine_legacy = make_case()
disabled_engine_legacy["servingEngineSpec"]["enableEngine"] = False
disabled_engine_legacy["servingEngineSpec"]["modelSpec"]["pd"]["ucm"] = False
cases["disabled-engine-legacy"] = disabled_engine_legacy

for name, values in cases.items():
    with open(os.path.join(work, f"{name}.yaml"), "w", encoding="utf-8") as stream:
        yaml.safe_dump(values, stream, allow_unicode=True, sort_keys=False)
PYEOF

render_ok() {
  local case_name="$1"
  echo "   render ${case_name}"
  helm template rel . -f "$WORK/${case_name}.yaml" > "$WORK/render-${case_name}.yaml"
}

expect_fail() {
  local case_name="$1"
  shift
  local output="$WORK/fail-${case_name}.log"
  if helm template rel . -f "$WORK/${case_name}.yaml" >"$output" 2>&1; then
    echo "   FAIL: ${case_name} 应渲染失败却成功"
    exit 1
  fi
  local pattern
  for pattern in "$@"; do
    if ! grep -Eqi -- "$pattern" "$output"; then
      echo "   FAIL: ${case_name} 的报错未包含预期信息: ${pattern}"
      sed -n '1,120p' "$output"
      exit 1
    fi
  done
}

echo "== 1) V1 / Hybrid UCM on/off + NIXL off 渲染矩阵 =="
for case_name in v1-off v1-on hybrid-off hybrid-on nixl-off v1-default-on v1-alias-on native-offloading-no-chart-kv kv-dp-env-sanitized; do
  render_ok "$case_name"
done

python3 - "$WORK" <<'PYEOF'
import json
import os
import re
import sys

import yaml


work = sys.argv[1]
case_specs = {
    "v1-off": ("MooncakeConnectorV1", "mooncake", False),
    "v1-on": ("MooncakeConnectorV1", "mooncake", True),
    "hybrid-off": ("MooncakeHybridConnector", "mooncake", False),
    "hybrid-on": ("MooncakeHybridConnector", "mooncake", True),
    "nixl-off": ("NixlConnector", "nixl", False),
    "v1-default-on": ("MooncakeConnectorV1", "mooncake", True),
    "v1-alias-on": ("MooncakeConnectorV1", "mooncake", True),
}


def docs_for(case_name):
    path = os.path.join(work, f"render-{case_name}.yaml")
    return [doc for doc in yaml.safe_load_all(open(path, encoding="utf-8")) if doc]


def values_for_key(value, wanted):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == wanted:
                result.append(child)
            result.extend(values_for_key(child, wanted))
    elif isinstance(value, list):
        for child in value:
            result.extend(values_for_key(child, wanted))
    return result


def values_for_any_key(value, wanted):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in wanted:
                result.append(child)
            result.extend(values_for_any_key(child, wanted))
    elif isinstance(value, list):
        for child in value:
            result.extend(values_for_any_key(child, wanted))
    return result


def connector_children(config):
    extra = config.get("kv_connector_extra_config") or {}
    children = extra.get("connectors") or []
    assert isinstance(children, list), children
    return children


def assert_dynamic_sentinel(value, field):
    assert isinstance(value, str), f"{field} sentinel 应为 JSON 字符串，实际 {value!r}"
    assert value and not value.isdecimal(), f"{field} 仍是静态身份值: {value!r}"


def configmap_sources(pod_spec):
    volume = next(v for v in pod_spec["volumes"] if v["name"] == "entrypoint-config")
    result = []
    for source in volume["projected"]["sources"]:
        if "configMap" in source:
            result.append(source["configMap"]["name"])
    return result


def env_field_paths(container):
    result = {}
    for item in container.get("env", []):
        field_ref = (item.get("valueFrom") or {}).get("fieldRef")
        if field_ref:
            result[item["name"]] = field_ref.get("fieldPath")
    return result


def assert_identity_meta(meta, connector, role):
    blob = json.dumps(meta, ensure_ascii=False)
    assert role in blob, f"meta 未记录 role {role}: {meta}"
    engine_bases = values_for_any_key(meta, {"engineIdBase", "engine_id_base"})
    assert 0 in engine_bases, f"meta 缺 engineIdBase=0: {meta}"
    assert "WORKER_INDEX" not in blob, "worker index 不得参与逻辑实例身份"
    if connector.startswith("Mooncake"):
        port_bases = values_for_any_key(meta, {"kvPortBase", "kv_port_base"})
        strides = values_for_any_key(meta, {"instanceStride", "instance_stride"})
        assert 36000 in port_bases, f"meta 缺 kvPortBase=36000: {meta}"
        assert 100 in strides, f"meta 缺 instanceStride=100: {meta}"
    else:
        assert not values_for_any_key(meta, {"kvPortBase", "kv_port_base"}), meta
        assert not values_for_any_key(meta, {"instanceStride", "instance_stride"}), meta
        assert not values_for_any_key(meta, {"portSpan", "port_span"}), meta


def assert_template(config, connector, role, ucm_on):
    engine_values = values_for_key(config, "engine_id")
    port_values = values_for_key(config, "kv_port")
    assert len(engine_values) == 1, f"engine_id 必须且只能出现一次: {config}"
    assert_dynamic_sentinel(engine_values[0], "engine_id")

    expected_kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
    if connector == "NixlConnector":
        assert config["kv_connector"] == connector, config
        assert config["kv_role"] == expected_kv_role, config
        assert not port_values, f"NIXL 不得含 kv_port: {config}"
        assert not connector_children(config), config
        return

    assert len(port_values) == 1, f"Mooncake kv_port 必须且只能出现一次: {config}"
    assert_dynamic_sentinel(port_values[0], "kv_port")
    if ucm_on and role == "prefill":
        assert config["kv_connector"] == "MultiConnector", config
        assert config["kv_role"] == "kv_producer", config
        assert "engine_id" in config, "MultiConnector engine_id 必须位于根节点"
        children = connector_children(config)
        assert len(children) == 2, children
        by_name = {child["kv_connector"]: child for child in children}
        assert set(by_name) == {connector, "UCMConnector"}, by_name
        selected = by_name[connector]
        ucm = by_name["UCMConnector"]
        assert "engine_id" not in selected and "engine_id" not in ucm, children
        assert "kv_port" in selected and "kv_port" not in ucm, children
        assert ucm["kv_role"] == "kv_both", ucm
    else:
        assert config["kv_connector"] == connector, config
        assert config["kv_role"] == expected_kv_role, config
        assert "engine_id" in config and "kv_port" in config, config
        assert not connector_children(config), config
        selected = config

    parallel = selected["kv_connector_extra_config"]
    assert parallel["prefill"] == {"dp_size": 2, "tp_size": 1, "pp_size": 2}, parallel
    assert parallel["decode"] == {"dp_size": 2, "tp_size": 1, "pp_size": 1}, parallel

    # V1 的 rank 是传输角色，不是实例编号；Hybrid 不得继承 V1 专属字段。
    if connector == "MooncakeConnectorV1":
        expected_rank = 0 if role == "prefill" else 1
        assert selected.get("kv_rank") == expected_rank, selected
    else:
        assert not values_for_key(config, "kv_rank"), f"Hybrid 泄漏 V1 kv_rank: {config}"


for case_name, (connector, router_type, ucm_on) in case_specs.items():
    docs = docs_for(case_name)
    cms = {doc["metadata"]["name"]: doc for doc in docs if doc.get("kind") == "ConfigMap"}
    model_servers = [doc for doc in docs if doc.get("kind") == "ModelServer"]
    model_servings = [doc for doc in docs if doc.get("kind") == "ModelServing"]
    assert len(model_servers) == 1 and len(model_servings) == 1

    # ModelServer 只消费 routerType；vLLM 原始 connector 名只进入 KV JSON。
    model_server = model_servers[0]
    assert model_server["spec"]["kvConnector"]["type"] == router_type, model_server
    assert connector not in json.dumps(model_server), model_server

    resolver_cms = [
        name
        for name, cm in cms.items()
        if "resolve-kv-transfer-config.py" in (cm.get("data") or {})
    ]
    assert len(resolver_cms) == 1, f"共享 resolver 应且只应渲染一次: {resolver_cms}"
    resolver_cm = resolver_cms[0]
    resolver = cms[resolver_cm]["data"]["resolve-kv-transfer-config.py"]
    assert not re.search(
        r"(?:environ(?:\.get)?|getenv)\s*(?:\[|\()\s*['\"]WORKER_INDEX['\"]",
        resolver,
    ), "resolver 不得从环境读取 WORKER_INDEX 计算身份"

    shared = cms["rel-vllm-args"]["data"]
    entrypoint = shared["args-entrypoint.sh"]
    active_lines = [
        line
        for line in entrypoint.splitlines()
        if not line.lstrip().startswith("#")
    ]
    active_entrypoint = "\n".join(active_lines)
    common_entrypoint = cms["rel-entrypoint-common"]["data"]["common-entrypoint.sh"]
    assert 'export VLLM_ARGS_FILE="${ENTRYPOINT_DIR}/vllm.args"' in common_entrypoint
    assert 'VLLM_ARGS_FILE:-' not in common_entrypoint
    assert "readonly ENTRYPOINT_DIR" in common_entrypoint
    lock_pos = active_entrypoint.index(
        "readonly UC_PD_GROUP_NAME UC_PD_ROLE_ID UC_USES_UCM"
    )
    prepare_pos = active_entrypoint.index("prepare_common_runtime", lock_pos)
    args_lock_pos = active_entrypoint.index("readonly VLLM_ARGS_FILE", prepare_pos)
    hook_pos = active_entrypoint.find('source "${ENTRYPOINT_DIR}/hook-pre-start.sh"')
    if hook_pos != -1:
        assert lock_pos < prepare_pos < args_lock_pos < hook_pos, active_entrypoint
    append_matches = re.findall(
        r"cmd\s*\+=\s*\([^)]*--kv-transfer-config",
        active_entrypoint,
        re.DOTALL,
    )
    assert len(append_matches) == 1, "--kv-transfer-config 必须且只能通过一次 cmd+=(...) 追加"
    assert re.search(
        r'option_name="\$\{option_name//_/-\}".*"\$\{managed\}" == "\$\{option_name\}"\*',
        active_entrypoint,
        re.DOTALL,
    ), "解析 vllmArgs 后必须按 vLLM underscore/abbreviation 语义拒绝托管 flag"
    assert "--output \"${output_file}\"" in active_entrypoint, active_entrypoint
    assert not re.search(
        r'resolved="\$\(python3[^\n]*resolve-kv-transfer-config',
        active_entrypoint,
    ), "runtime 不得再用 resolver stdout 作为 JSON 数据通道"
    for env_name in (
        "VLLM_DP_SIZE",
        "VLLM_DP_RANK",
        "VLLM_DP_RANK_LOCAL",
        "VLLM_DP_MASTER_IP",
        "VLLM_DP_MASTER_PORT",
    ):
        assert env_name in active_entrypoint, env_name
    sanitize_pos = active_entrypoint.index('unset "${kv_env_name}"')
    cmd_pos = active_entrypoint.index("cmd=(", sanitize_pos)
    assert sanitize_pos < cmd_pos, active_entrypoint

    model_roles = {
        role["name"]: role
        for role in model_servings[0]["spec"]["template"]["roles"]
    }
    for role in ("prefill", "decode"):
        cm_name = f"rel-{role}-vllm-args"
        data = cms[cm_name]["data"]
        assert "--kv-transfer-config" not in data["vllm.args"], data["vllm.args"]
        assert "kv-transfer.template.json" in data, data.keys()
        assert "kv-transfer.meta.json" in data, data.keys()
        template = json.loads(data["kv-transfer.template.json"])
        meta = json.loads(data["kv-transfer.meta.json"])
        assert isinstance(template, dict) and isinstance(meta, dict)
        assert_template(template, connector, role, ucm_on)
        assert_identity_meta(meta, connector, role)

        # entry 与 worker 都从同一个 role ConfigMap 读取相同模板/元数据，身份只由
        # group-name + role-id 推导。
        for template_name in ("entryTemplate", "workerTemplate"):
            pod_spec = model_roles[role][template_name]["spec"]
            container = pod_spec["containers"][0]
            fields = env_field_paths(container)
            assert fields.get("UC_PD_GROUP_NAME") == (
                "metadata.labels['modelserving.volcano.sh/group-name']"
            ), fields
            assert fields.get("UC_PD_ROLE_ID") == (
                "metadata.labels['modelserving.volcano.sh/role-id']"
            ), fields
            sources = configmap_sources(pod_spec)
            assert cm_name in sources and resolver_cm in sources, sources

            volume_names = {volume["name"] for volume in pod_spec.get("volumes", [])}
            env_by_name = {item["name"]: item for item in container.get("env", [])}
            env_names = set(env_by_name)
            role_uses_ucm = ucm_on and role == "prefill"
            assert env_by_name["UC_USES_UCM"].get("value") == str(role_uses_ucm).lower(), env_by_name
            assert ("ucm-config-runtime" in volume_names) == role_uses_ucm, volume_names
            assert ("DATA_DIRS" in env_names) == role_uses_ucm, env_names

    assert ("rel-ucm-config" in cms) == ucm_on, cms.keys()

print("   PASS")
PYEOF

echo "== 2) 非法 connector/router/UCM 组合必须 Helm fail =="
expect_fail nixl-on 'NixlConnector|nixl' 'UCM|unifiedcacheConfig'
expect_fail router-mismatch 'routerType' 'MooncakeConnectorV1'
expect_fail unknown-connector 'UnknownConnector' 'connector'
expect_fail connector-case-error 'mooncakeconnectorv1' 'connector'
expect_fail nixl-mooncake-master 'mooncake.*master|master.*mooncake' 'routerType|mooncake'
expect_fail missing-connector 'connector.*required|required.*connector'
expect_fail missing-router 'routerType.*required|required.*routerType'
expect_fail managed-MultiConnector 'MultiConnector' 'chart-managed|cannot'
expect_fail managed-UCMConnector 'UCMConnector' 'chart-managed|cannot'
expect_fail missing-engine 'engineIdBase' 'required'
expect_fail unknown-identity 'identity' 'unsupported|engine_id'
expect_fail negative-engine 'engineIdBase' 'integer|>= 0'
expect_fail fractional-engine 'engineIdBase' 'integer'
expect_fail string-port 'kvPortBase' 'integer'
expect_fail fractional-port 'kvPortBase' 'integer'
expect_fail missing-port 'kvPortBase' 'required'
expect_fail fractional-stride 'instanceStride' 'integer'
expect_fail small-stride 'instanceStride' '100|too small'
expect_fail parallel-stride 'instanceStride' '256|parallel|too small'
expect_fail decode-pp 'decode.*pipeline parallel|pipeline parallel.*decode' 'must be 1'
expect_fail managed-arg-inline 'kv-transfer-config' 'chart-managed|must not set'
expect_fail managed-arg-quoted 'kv-transfer-config' 'chart-managed|must not set'
expect_fail managed-arg-underscore 'kv.transfer.config' 'chart-managed|must not set'
expect_fail managed-arg-abbrev 'kv-transfer-config' 'chart-managed|must not set'
expect_fail managed-vllm-config 'config' 'chart-managed|must not set'
expect_fail managed-arg-dotted 'kv-transfer-config' 'chart-managed|must not set'
for alias in dpa dpp dpr; do
  expect_fail "managed-short-${alias}" "-${alias}" 'chart-managed|must not set'
done
expect_fail layout-abbrev 'tensor-parallel-size' 'full|abbreviation'
expect_fail kv-offloading-size 'kv-offloading-size' 'chart-managed|must not set'
expect_fail kv-offloading-backend 'kv-offloading-backend' 'chart-managed|must not set'
for managed_env in uc-pd-group-name uc-pd-role-id uc-uses-ucm uc-skip-kv-connector-registry-probe vllm-args-file; do
  expect_fail "managed-env-${managed_env}" 'chart-managed' 'UC_|KV-transfer'
  expect_fail "managed-topology-env-${managed_env}" 'nodeTopologyConfig' 'chart-managed'
done
expect_fail topology-shell-key 'nodeTopologyConfig' 'invalid environment-variable name'
expect_fail disabled-engine-legacy 'pd\.ucm' 'unifiedcacheConfig|removed'
expect_fail overflow-port '65535|exceeds'
expect_fail nixl-port-fields 'kvPortBase|instanceStride' 'not allowed'
expect_fail same-roles 'different roles|prefill.*decode'
expect_fail unknown-role 'existing roles|roles.*name'
expect_fail empty-ucm 'config.*empty|empty.*config'
for switch_mode in default enabled alias; do
  expect_fail "invalid-ucm-config-${switch_mode}" 'unifiedcacheConfig\.config' 'must be a map'
done
expect_fail invalid-ucm-switch 'unifiedcacheConfig\.enabled' 'boolean'
echo "   PASS"

echo "== 3) 三个旧 pd 字段即使 false/空值也必须 Helm fail =="
for field in connector mooncakePort ucm; do
  for index in 0 1; do
    expect_fail "legacy-${field}-${index}" "pd\.${field}" 'kvTransfer|已删除|removed|迁移'
  done
done
echo "   PASS"

echo "ALL KV TRANSFER TESTS PASSED"
