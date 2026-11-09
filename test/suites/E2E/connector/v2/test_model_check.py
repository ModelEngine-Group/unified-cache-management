"""Opt-in E2E case. The checker declares hardware only in its own process."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).parent / "model_check" / "run.py"


def local_model(platform, model_id):
    """Read the pre-provisioned model snapshot without network access."""
    root = os.environ.get("UCM_V2_MODEL_ROOT")
    if not root:
        pytest.fail("Set UCM_V2_MODEL_ROOT to the provisioned model metadata directory")
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        pytest.fail(f"Model metadata manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    key = f"{platform}/{model_id}"
    if key not in manifest:
        pytest.fail(f"Model metadata manifest has no {key} entry")
    model = root / platform / model_id
    files = manifest[key].get("files_sha256", {})
    if "config.json" not in files:
        pytest.fail(f"Model metadata manifest has no config hash for {key}")
    for filename, expected in files.items():
        path = model / filename
        if not path.is_file():
            pytest.fail(f"Model metadata file is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            pytest.fail(f"Model metadata hash mismatch: {path}")
    return model


def hybrid_platform_args(platform, suite="HYBRID"):
    """Opt in separately to each fixed platform, retaining the old launcher."""
    env_name = f"UCM_V2_{suite}_{platform.upper()}_ARGS"
    if env_name in os.environ:
        args = shlex.split(os.environ[env_name])
        if any(arg == "--platform" or arg.startswith("--platform=") for arg in args):
            raise ValueError(f"{env_name} must not override the test platform")
        target = (
            ["--platform", "cuda-sim", "--cuda-capability", "9.0"]
            if platform == "cuda"
            else ["--platform", "ascend-sim", "--ascend-soc", "A2"]
        )
        return [*target, *args]

    # Existing callers may continue selecting one platform with the shared
    # variable. The other parametrized platform stays explicitly skipped.
    legacy = os.environ.get("UCM_V2_HYBRID_ARGS") if suite == "HYBRID" else None
    if legacy:
        args = shlex.split(legacy)
        selected = [
            args[i + 1] if arg == "--platform" else arg.split("=", 1)[1]
            for i, arg in enumerate(args)
            if (arg == "--platform" and i + 1 < len(args))
            or arg.startswith("--platform=")
        ]
        if len(selected) != 1 or selected[0] not in ("cuda-sim", "ascend-sim"):
            raise ValueError("UCM_V2_HYBRID_ARGS must select exactly one platform")
        if selected[0] == f"{platform}-sim":
            return args
    pytest.skip(f"Set {env_name} to enable the {platform} cases")


# Fixed plan-check profiles reflect the recorded CPU/meta runs, not complete
# official deployment topologies. TP and the requested service block are
# inputs: native allocation and UCM may choose different effective sizes.
HYBRID_PROFILES = {
    "ascend": {
        "deepseek-v4-flash": (4, None),
        "deepseek-v4.1-flash": (4, 128),
        "glm-5.3": (4, 128),
        "glm-5.3-flash": (8, 128),
        "qwen3.8-flash-next": (4, 1536),
        "qwen3.8-27b": (2, 128),
        "kimi-k3": (8, 128),
    },
    "cuda": {
        "deepseek-v4-flash": (4, 128),
        "deepseek-v4.1-flash": (4, 128),
        "glm-5.3": (4, 128),
        "glm-5.3-flash": (4, 128),
        "qwen3.8-flash-next": (4, 128),
        "qwen3.8-27b": (4, 128),
        "kimi-k3": (8, 768),
    },
}

# Speculator cases may have different target checkpoints and TP settings.
SPEC_PROFILES = {
    "cuda-dsv4-dspark": (8, 128),
    "cuda-dsv4-mtp": (4, 128),
    "ascend-dsv4-dspark": (8, 128),
    "ascend-dsv4-mtp": (4, None),
    "cuda-v41-dspark": (4, 128),
    "ascend-v41-dspark": (4, 128),
    "cuda-glm53-mtp": (8, 128),
    "ascend-glm53-mtp": (8, 128),
    "cuda-glm53flash-mtp": (4, 128),
    "ascend-glm53flash-mtp": (8, 128),
    "cuda-qwenfn-mtp": (4, 128),
    "ascend-qwenfn-mtp": (8, 1536),
    "cuda-qwen27-mtp": (1, 128),
    "ascend-qwen27-mtp": (2, 128),
    "cuda-kimi-dspark": (8, 768),
    "ascend-kimi-dspark": (4, 128),
}


def profile_args(profile):
    tp, service_block = profile
    args = [
        "--tp", str(tp), "--pcp", "1", "--dcp", "1",
        "--tokens", "32K", "--max-num-batched-tokens", "8K",
    ]
    if service_block is not None:
        args += ["--block-size", str(service_block)]
    return args


def fixed_case_args(args):
    # Do not silently change the configuration associated with a case ID.
    # The generic test_model_check entry remains available for custom probes.
    fixed = {
        "--tp", "--pcp", "--dcp", "--block-size", "--chunk-size",
        "--tokens", "--max-num-batched-tokens", "--kv-mib", "--dtype",
        "--quantization", "--kv-cache-dtype", "--indexer-kv-dtype",
        "--mamba-ssm-cache-dtype", "--model", "--speculative-config",
        "--assume-hit-percent", "--expect-policy", "--ucm-config",
    }
    for arg in args:
        if arg.split("=", 1)[0] in fixed:
            raise ValueError(
                f"{arg} changes a fixed model-check profile; use "
                "UCM_V2_MODEL_CHECK_ARGS for a custom run"
            )
    return args


def target_defaults(platform, model_id):
    if platform == "ascend" and model_id in ("kimi-k3", "qwen3.8-flash-next"):
        return ["--quantization", "ascend", "--trust-remote-code"]
    if platform == "cuda" and model_id in ("kimi-k3", "qwen3.8-flash-next"):
        defaults = ["--kv-cache-dtype", "fp8"]
        if model_id == "qwen3.8-flash-next":
            defaults.extend(["--indexer-kv-dtype", "fp8"])
        else:
            defaults.extend(["--mamba-ssm-cache-dtype", "bfloat16"])
        return defaults
    return []


# Native persistent policies each target must exercise. V4.1 rebuilds its
# sliding window through bounded replay, so only its full-attention groups are
# persisted.
MODEL_POLICIES = {
    "deepseek-v4-flash": ("FA", "WA"),
    "deepseek-v4.1-flash": ("FA",),
    "glm-5.3": ("FA",),
    "glm-5.3-flash": ("FA", "STATE"),
    "qwen3.8-flash-next": ("FA", "STATE"),
    "qwen3.8-27b": ("FA", "STATE"),
    "kimi-k3": ("FA", "STATE"),
}

# Official speculative configurations, one entry per target x platform x
# documented variant. Ascend values follow the vLLM-Ascend v0.32.0rc
# tutorials (Atlas 800I A2 tab where one exists); CUDA values follow
# recipes.vllm.ai for the vLLM 0.32 line. "draft" marks the only target whose
# draft is a separate checkpoint. A variant stays in this table when its
# checkpoint is not provisioned offline, so the failure is recorded instead of
# silently dropped.
SPECULATIVE = {
    # DeepSeek-V4-Flash: the published DSpark draft ships inside the 0731
    # checkpoint; the preview checkpoint (provisioned here) carries an MTP head.
    "cuda-dsv4-dspark": {
        "platform": "cuda",
        "model": "deepseek-v4-flash",
        "method": "dspark",
        "tokens": 7,
    },
    "cuda-dsv4-mtp": {
        "platform": "cuda",
        "model": "deepseek-v4-flash",
        "method": "mtp",
        "tokens": 3,
    },
    "ascend-dsv4-dspark": {
        "platform": "ascend",
        "model": "deepseek-v4-flash",
        "method": "dspark",
        "tokens": 7,
    },
    "ascend-dsv4-mtp": {
        "platform": "ascend",
        "model": "deepseek-v4-flash",
        "method": "mtp",
        "tokens": 3,
    },
    "cuda-v41-dspark": {
        "platform": "cuda",
        "model": "deepseek-v4.1-flash",
        "method": "dspark",
        "tokens": 5,
    },
    "ascend-v41-dspark": {
        "platform": "ascend",
        "model": "deepseek-v4.1-flash",
        "method": "dspark",
        "tokens": 5,
    },
    "cuda-glm53-mtp": {
        "platform": "cuda",
        "model": "glm-5.3",
        "method": "mtp",
        "tokens": 5,
    },
    "ascend-glm53-mtp": {
        "platform": "ascend",
        "model": "glm-5.3",
        "method": "deepseek_mtp",
        "tokens": 3,
    },
    "cuda-glm53flash-mtp": {
        "platform": "cuda",
        "model": "glm-5.3-flash",
        "method": "mtp",
        "tokens": 5,
    },
    "ascend-glm53flash-mtp": {
        "platform": "ascend",
        "model": "glm-5.3-flash",
        "method": "deepseek_mtp",
        "tokens": 3,
    },
    "cuda-qwenfn-mtp": {
        "platform": "cuda",
        "model": "qwen3.8-flash-next",
        "method": "mtp",
        "tokens": 3,
    },
    "ascend-qwenfn-mtp": {
        "platform": "ascend",
        "model": "qwen3.8-flash-next",
        "method": "qwen3_5_mtp",
        "tokens": 3,
    },
    "cuda-qwen27-mtp": {
        "platform": "cuda",
        "model": "qwen3.8-27b",
        "method": "mtp",
        "tokens": 3,
    },
    "ascend-qwen27-mtp": {
        "platform": "ascend",
        "model": "qwen3.8-27b",
        "method": "qwen3_5_mtp",
        "tokens": 3,
    },
    "cuda-kimi-dspark": {
        "platform": "cuda",
        "model": "kimi-k3",
        "method": "dspark",
        "tokens": 7,
        "draft": True,
    },
    "ascend-kimi-dspark": {
        "platform": "ascend",
        "model": "kimi-k3",
        "method": "dspark",
        "tokens": 7,
        "draft": True,
    },
}


def speculative_config(case):
    """Official method/token count plus the platform's eager draft switch."""
    spec = {"method": case["method"], "num_speculative_tokens": case["tokens"]}
    if case.get("draft"):
        # Kimi-K3 drafts with a separate checkpoint; every other target ships
        # its MTP/DSpark stages inside the target weights.
        spec["model"] = str(local_model(case["platform"], "kimi-k3-dspark-inferact"))
    if case["platform"] == "ascend":
        spec["enforce_eager"] = True
    return spec


def offline_env():
    return {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MODELSCOPE_OFFLINE": "1",
    }


@pytest.mark.stage(2)
@pytest.mark.feature("connector")
def test_model_check():
    args = os.environ.get("UCM_V2_MODEL_CHECK_ARGS")
    if not args:
        pytest.skip("Set UCM_V2_MODEL_CHECK_ARGS to the run.py arguments")
    subprocess.run([sys.executable, str(RUNNER), *shlex.split(args)], check=True)


@pytest.mark.stage(2)
@pytest.mark.feature("connector")
@pytest.mark.parametrize(("model_id", "policies"), MODEL_POLICIES.items())
@pytest.mark.parametrize("hit", [0, 90], ids=["cold", "hit90"])
@pytest.mark.parametrize("platform", ["cuda", "ascend"])
def test_hybrid_model(model_id, policies, hit, platform):
    args = fixed_case_args(hybrid_platform_args(platform))
    model = local_model(platform, model_id)
    defaults = target_defaults(platform, model_id)
    profile = profile_args(HYBRID_PROFILES[platform][model_id])
    print(f"Model: {platform}/{model_id} ({model})", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            *args,
            *defaults,
            "--kv-mib",
            "16384",
            *profile,
            "--model",
            str(model),
            "--assume-hit-percent",
            str(hit),
            "--expect-policy",
            *policies,
        ],
        check=True,
        env=offline_env(),
    )


@pytest.mark.stage(2)
@pytest.mark.feature("connector")
@pytest.mark.parametrize(
    ("case_id", "case"), SPECULATIVE.items(), ids=SPECULATIVE.keys()
)
@pytest.mark.parametrize("hit", [0, 90], ids=["cold", "hit90"])
def test_speculative_layout(case_id, case, hit):
    """Inspect target and draft KV plans; this does not execute verification."""
    platform = case["platform"]
    model_id = case["model"]
    args = fixed_case_args(hybrid_platform_args(platform, suite="SPEC"))
    snapshot = "deepseek-v4-flash-0731" if case_id.endswith("dsv4-dspark") else model_id
    model = local_model(platform, snapshot)
    spec = speculative_config(case)
    defaults = target_defaults(platform, model_id)
    if platform == "cuda" and model_id in ("deepseek-v4-flash", "deepseek-v4.1-flash"):
        defaults += ["--kv-cache-dtype", "fp8"]
    print(f"Model: {platform}/{snapshot} ({model}); case={case_id}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            *args,
            *defaults,
            "--kv-mib",
            "16384",
            *profile_args(SPEC_PROFILES[case_id]),
            "--model",
            str(model),
            "--speculative-config",
            json.dumps(spec),
            "--assume-hit-percent",
            str(hit),
            "--expect-policy",
            *MODEL_POLICIES[model_id],
        ],
        check=True,
        env=offline_env(),
    )
