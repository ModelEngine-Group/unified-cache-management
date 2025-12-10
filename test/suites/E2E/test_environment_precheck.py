from pathlib import Path

import pytest
import yaml
from common.envPreCheck.run_env_preCheck import (
    LOCAL_SSH_KEY,
    MASTER_IP,
    WORKER_IP,
    run_bandwidth_check,
    run_check_hccn_ping,
    run_check_model_weight,
    run_check_nvidia_ping,
    run_hccn_device_status_check,
    run_hccn_tls_check,
    run_nvidia_device_status_check,
    run_set_ssh_login,
)


# ========= Environment Setup =========
@pytest.mark.stage(1)
@pytest.mark.platform("npu")
@pytest.mark.feature("test_ssh_login")
def test_ssh_login():
    """
    Load environment variables from file and configure SSH login.
    Dynamically generates value_lists based on ssh_result content.

    Returns:
        dict: SSH check result in value_lists format
    """
    ssh_result = run_set_ssh_login()

    value_lists = {}

    # Dynamically iterate through ssh_result to generate value_lists
    for key, value in ssh_result.items():
        # Structure like {"MASTER_IP": {"ip": "...", "status": True}}
        if isinstance(value, dict) and "status" in value:
            value_lists[f"SSH_CHECK_{key}"] = value["status"]

        # Structure like {"SSH_CHECK": True}
        elif isinstance(value, bool):
            value_lists[key] = value

        else:
            continue

    # Ensure SSH_CHECK itself must pass
    assert ssh_result.get("SSH_CHECK", False) is True, f"SSH login failed: {ssh_result}"


# ========= Device Status Check =========
@pytest.mark.stage(1)
@pytest.mark.platform("npu")
@pytest.mark.feature("test_check_hccn_device_status")
def test_hccn_check_device_status():
    """
    Check the status of all devices in the cluster.
    Dynamically generates value_lists based on device_status content.

    Returns:
        dict: Device status check result in value_lists format
    """
    device_status = run_hccn_device_status_check()

    value_lists = {}

    # Dynamically iterate through device_status, add to value_lists when status is found
    for key, value in device_status.items():
        if isinstance(value, dict) and "status" in value:
            value_lists[key] = [value["status"]]

    # Overall validation
    assert (
        device_status.get("device_status_check", {}).get("status", False) is True
    ), f"Device status check failed: {device_status}"


@pytest.mark.stage(1)
@pytest.mark.platform("gpu")
@pytest.mark.feature("test_check_nvidia_device_status")
def test_nvidia_check_device_status():
    """
    Check the status of all devices in the cluster.
    Dynamically generates value_lists based on device_status content.

    Returns:
        dict: Device status check result in value_lists format
    """
    device_status = run_nvidia_device_status_check()

    value_lists = {}

    # Dynamically iterate through device_status, add to value_lists when status is found
    for key, value in device_status.items():
        if isinstance(value, dict) and "status" in value:
            value_lists[key] = [value["status"]]

    # Overall validation
    assert (
        device_status.get("device_status_check", {}).get("status", False) is True
    ), f"Device status check failed: {device_status}"


# ========= Ping Connectivity Check =========
@pytest.mark.stage(1)
@pytest.mark.platform("npu")
@pytest.mark.feature("test_check_ascend_ping")
def test_check_hccn_ping():
    """
    Perform ping connectivity check for the detected environment type (ASCEND).

    Returns:
        dict: Ping check result with dynamic value_lists
    """
    ping_result = run_check_hccn_ping()

    # Dynamic value_lists container
    value_lists = {}

    # 1Add all keys except the summary key (hccn_ping_check)
    for key, info in ping_result.items():
        if key == "hccn_ping_check":
            continue
        value_lists[key] = [info.get("status", False)]

    # Add final summary ping check key as the last entry
    final_status = ping_result.get("hccn_ping_check", {}).get("status", False)
    value_lists["hccn_ping_check"] = [final_status]

    # Assertion: final summary must pass
    assert final_status is True, f"ASCEND ping failed: {ping_result}"


@pytest.mark.stage(1)
@pytest.mark.platform("gpu")
@pytest.mark.feature("test_check_nvidia_ping")
def test_check_nvidia_ping():
    """
    Perform ping connectivity check for the detected environment type (NVIDIA).

    Returns:
        dict: Ping check result with dynamic value_lists
    """
    ping_result = run_check_nvidia_ping()

    # Dynamic value_lists container
    value_lists = {}

    # 1Add all keys except the summary key (nvidia_ping_check)
    for key, info in ping_result.items():
        if key == "nvidia_ping_check":
            continue
        value_lists[key] = [info.get("status", False)]

    # Add final summary ping check key as the last entry
    final_status = ping_result.get("nvidia_ping_check", {}).get("status", False)
    value_lists["nvidia_ping_check"] = [final_status]

    # Assertion: final summary must pass
    assert final_status is True, f"NVIDIA ping failed: {ping_result}"


# ========= TLS Connectivity Check =========
@pytest.mark.stage(1)
@pytest.mark.platform("npu")
@pytest.mark.feature("test_check_tls")
def test_check_tls():
    """
    Check TLS switch status of all HCCN cards on master and worker nodes.
    Dynamically build value_lists based on tls_result content.

    Returns:
        dict: TLS check result stored inside value_lists
    """
    tls_result = run_hccn_tls_check(MASTER_IP, WORKER_IP)

    tls_data = tls_result.get("tls_check", {})
    value_lists = {}

    # Dynamically extract each field and its status
    for key, entry in tls_data.items():
        if key == "status":
            continue
        else:
            # Each card's status
            value_lists[key] = [entry.get("status", False)]

    value_lists["tls_check"] = [tls_data.get("status", False)]

    # Assert overall TLS status
    assert tls_data.get("status", False) is True, f"TLS check failed: {tls_result}"


# ========= Model Weight Files Check =========
@pytest.mark.stage(1)
@pytest.mark.platform("npu")
@pytest.mark.feature("test_check_model_weights")
def test_check_model_weights():
    """
    Verify the existence and correctness of model weight files.

    Returns:
        dict: Model weight check result with dynamic value_lists
    """

    model_weight = run_check_model_weight()

    # Dynamic value_lists container
    value_lists = {}

    # Iterate through all top-level keys except "model_weight_check"
    for key, info in model_weight.items():
        if key == "model_weight_check":
            continue
        # Add status as list
        value_lists[key] = [info.get("status", False)]

    # Add model_weight_check as last field
    final_status = model_weight.get("model_weight_check", {}).get("status", False)
    value_lists["model_weight_check"] = [final_status]

    # Final assertion
    assert final_status is True, f"Model weight check failed: {model_weight}"


# ========= Bandwidth Check =========
@pytest.mark.stage(1)
@pytest.mark.platform("npu")
@pytest.mark.feature("test_check_bandwidth")
def test_check_bandwidth():
    """
    Measure storage system bandwidth for embedding and fetching operations.
    Dynamically generate value_lists from the returned bandwidth dict.

    Returns:
        dict: Bandwidth summary stored in value_lists format
    """
    bandwidth = run_bandwidth_check()
    print("bandwidth", bandwidth)

    value_lists = {}

    # Dynamically generate key, e.g., embed → embed_avg_bandwidth
    for key, val in bandwidth.items():
        dynamic_key = f"{key}_avg_bandwidth"
        value_lists[dynamic_key] = [val]

    config_file = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        EXPECTED_EMBED_BANDWIDTH = config.get("Env_preCheck", {}).get(
            "expected_embed_bandwidth", ""
        )
        EXPECTED_FETCH_BANDWIDTH = config.get("Env_preCheck", {}).get(
            "expected_fetch_bandwidth", ""
        )

    # Validation
    assert (
        bandwidth["embed"] < 0.85 * EXPECTED_EMBED_BANDWIDTH
    ), f"Embed bandwidth too high: {bandwidth['embed']} GB/s"
    assert (
        bandwidth["fetch"] < 0.85 * EXPECTED_FETCH_BANDWIDTH
    ), f"Fetch bandwidth too high: {bandwidth['fetch']} GB/s"
