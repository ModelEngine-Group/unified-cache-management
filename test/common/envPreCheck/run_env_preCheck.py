import mmap
import multiprocessing
import os
import re
import secrets
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from ucm.store.factory_v1 import UcmConnectorFactoryV1, UcmKVStoreBaseV1

CODE_ROOT = Path(__file__).resolve().parent
Custom_SSH_DIR = (CODE_ROOT / "ssh_keys").resolve()
Custom_SSH_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_SSH_KEY = Custom_SSH_DIR / "id_rsa"
LOCAL_SSH_KEY_PUB = Custom_SSH_DIR / "id_rsa.pub"

config_file = Path(__file__).parent.parent.parent / "config.yaml"
with open(config_file, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    MASTER_IP = config.get("Env_preCheck", {}).get("master_ip", "")
    WORKER_IP = config.get("Env_preCheck", {}).get("worker_ip", "")
    ASCEND_RT_VISIBLE_DEVICES = config.get("Env_preCheck", {}).get(
        "ascend_rt_visible_devices", ""
    )
    NODE_NUM = config.get("Env_preCheck", {}).get("node_num", "")
    MODEL_PATH = config.get("Env_preCheck", {}).get("model_path", "")
    HF_MODEL_NAME = config.get("Env_preCheck", {}).get("hf_model_name", "")
    MIDDLE_PAGE = config.get("Env_preCheck", {}).get("middle_page", "")

    WORKER_NUMBER = config.get("Env_preCheck", {}).get("worker_number", 1)
    SHARD_SIZE = config.get("Env_preCheck", {}).get("shard_size", 1 * 1024 * 1024)
    SHARD_NUMBER = config.get("Env_preCheck", {}).get("shard_number", 1)
    BLOCK_NUMBER = config.get("Env_preCheck", {}).get("block_number", 64)
    DUMP_EPOCH_NUMBER = config.get("Env_preCheck", {}).get("dump_epoch_number", 32)
    LOAD_EPOCH_NUMBER = config.get("Env_preCheck", {}).get("load_epoch_number", 32)
    STORAGE_BACKENDS = config.get("Env_preCheck", {}).get("storage_backends", "")


def run_command(
    cmd: List[str], check: bool = True, timeout: Optional[int] = None
) -> Tuple[int, str, str]:
    """
    Execute a command and return (return_code, stdout, stderr).

    Args:
        cmd: List of command arguments.
        check: If True, raise an exception on non-zero return code.
        timeout: Maximum time (in seconds) to wait for command completion.
                 If None, no timeout is applied.

    Returns:
        A tuple containing:
        - return_code: Exit code of the command.
        - stdout: Captured standard output.
        - stderr: Captured standard error.

    Raises:
        subprocess.TimeoutExpired: If the command times out.
        Exception: For any other unexpected error.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return -1, "", str(e)


def print_info(msg: str):
    """Print an info-level message."""
    print(f"[INFO] {msg}")


def print_error(msg: str):
    """Print an error-level message to stderr."""
    print(f"[ERROR] {msg}", file=sys.stderr)


def import_env_from_file(config_path: Optional[Path] = None) -> List[str]:
    """
    Load environment variables from a properties file.

    This function reads a configuration file (default: 'config.properties')
    and exports each key-value pair as an environment variable.

    Args:
        config_path (Path, optional): Path to the config file.
            If not provided, defaults to 'config.properties' in the script's directory.

    Returns:
        List[str]: List of environment variable names that were successfully loaded.

    Raises:
        SystemExit: If the config file does not exist.
    """

    # Determine config file path
    config_file = config_path or CODE_ROOT / "config.properties"

    # Check if config file exists
    if not config_file.exists():
        print(f"Configuration file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    print("\n========================================")
    print("Loading environment variables from config file:")
    print("========================================")
    time.sleep(2)

    loaded_vars = []

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Ensure line has exactly one '='
                if line.count("=") != 1:
                    print(f"Invalid format on line {line_num}: {line}", file=sys.stderr)
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Skip if key is empty
                if not key:
                    print(f"Empty variable name on line {line_num}", file=sys.stderr)
                    continue

                # Set environment variable
                os.environ[key] = value
                loaded_vars.append(key)

                # Print exported variable
                print(f"export {key}={value}")

    except PermissionError:
        print(
            f"Permission denied when reading config file: {config_file}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error while reading config file: {e}", file=sys.stderr)
        sys.exit(1)

    print("========================================")
    print(f"Successfully loaded {len(loaded_vars)} environment variables.")
    print("========================================")
    time.sleep(2)


def check_ssh_login(ip: str) -> bool:
    """
    Test SSH passwordless login to a remote host.

    Args:
        ip: IP address of the remote host.

    Returns:
        True if SSH login succeeds without password prompt; False otherwise.
    """
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-i",
        str(LOCAL_SSH_KEY),
        f"root@{ip}",
        "exit",
    ]
    return run_command(cmd, check=False)[0] == 0


def setup_ssh_key(ip: str) -> None:
    """
    Configure passwordless SSH login for a remote host.

    Steps:
        1. Ensure local SSH key pair exists (generate if missing).
        2. Upload public key to the remote host using ssh-copy-id.
        3. Verify that passwordless SSH login works.

    Args:
        ip: IP address of the target remote host.

    Raises:
        SystemExit: If any step fails (e.g., key generation, upload, or verification).
    """
    print(f"[UC] Configuring passwordless SSH login for {ip}...")

    # Ensure local SSH key pair exists
    if not LOCAL_SSH_KEY.exists():
        print_info(f"Generating new SSH key pair: {LOCAL_SSH_KEY}")
        ret, _, err = run_command(
            [
                "ssh-keygen",
                "-t",
                "rsa",
                "-b",
                "4096",
                "-f",
                str(LOCAL_SSH_KEY),
                "-N",
                "",  # No passphrase
            ]
        )
        if ret != 0:
            print_error(f"Failed to generate SSH key: {err}")
            # sys.exit(1)
            return False

    # Upload public key to remote host
    print_info(f"Uploading public key to {ip} (please enter password manually)...")
    ret, out, err = run_command(["ssh-copy-id", "-i", str(LOCAL_SSH_KEY), f"root@{ip}"])

    if ret != 0:
        print_error(f"Failed to upload public key: {err}")
        return False

    # Verify passwordless SSH login
    if check_ssh_login(ip):
        print_info(f"Passwordless SSH login configured successfully for {ip}!")
    else:
        print_error(
            f"Failed to verify passwordless SSH login for {ip}. "
            "Please check network connectivity, SSH service, or user permissions."
        )
        # sys.exit(1)
        return False


def run_set_ssh_login() -> Dict[str, Any]:
    """
    Configure passwordless SSH login for master and worker nodes.

    This function checks and sets up SSH key-based authentication
    for the master and worker nodes defined via environment variables.

    Returns:
        A dictionary containing the configuration status for each node,
        along with a global success flag.

        Example structure:
        {
            "MASTER_IP": {
                "ip": "192.168.1.1",
                "status": True,
            },
            "WORKER_IP": {
                "ip": "192.168.1.2",
                "status": False,
            },
            "SSH_CHECK": True  # Overall success status
        }
    """
    print("\n" + "-" * 40 + "\n")
    print("Configuring passwordless SSH login for remote hosts")

    results: Dict[str, Dict[str, Any]] = {}
    all_success = True  # Track overall operation success

    # Build IP mapping
    ip_map: Dict[str, Optional[str]] = {"MASTER_IP": MASTER_IP}
    if WORKER_IP:
        ip_map["WORKER_IP"] = WORKER_IP

    # Process each target host
    for key, ip in ip_map.items():
        if not ip:
            result = {
                "ip": "",
                "status": False,
            }
            results[key] = result
            all_success = False
            continue

        result = {
            "ip": ip,
            "status": False,
        }

        print(f"\nProcessing IP: {ip}")

        try:
            # Step 1: Check if passwordless SSH is already working
            if check_ssh_login(ip):
                result["status"] = True
                print_info(f"{ip} - Passwordless login already available")
            else:
                # Step 2: Set up SSH key (will prompt for password)
                print_info(f"Setting up SSH key for {ip}...")
                setup_ssh_key(ip)

                # Step 3: Re-check after setup
                if check_ssh_login(ip):
                    result["status"] = True
                else:
                    result["status"] = False

        except Exception as e:
            result["status"] = False
            print_error(f"Error processing {ip}: {e}")

        # Update overall success flag
        if not result["status"]:
            all_success = False

        results[key] = result

    # Final global success indicator
    results["SSH_CHECK"] = all_success

    return results


def run_remote(ip, script, env=None, raise_on_error=True):
    """
    Execute a Bash script remotely via SSH, print output in real-time,
    and return the full output.

    Args:
        ip (str): Remote host IP address
        script (str): Bash script content to execute
        env (dict, optional): Environment variables to set before execution
        raise_on_error (bool): Whether to raise an exception if remote command fails

    Returns:
        str: Full output of the executed script

    Raises:
        Exception: If the remote command fails and raise_on_error=True
    """
    exports = ""
    if env:
        for k, v in env.items():
            exports += f'export {k}="{v}"\n'

    # Start SSH process with bash -s
    process = subprocess.Popen(
        ["ssh", "-q", "-i", str(LOCAL_SSH_KEY), f"root@{ip}", "bash", "-s"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Send environment exports and script
    process.stdin.write(exports + "\n" + script + "\n")
    process.stdin.close()

    # Read and print output line by line in real-time
    output_lines = []
    for line in process.stdout:
        print(line, end="")  # 实时打印
        output_lines.append(line)

    process.wait()
    full_output = "".join(output_lines)

    return full_output


def parse_hccn_output(output):
    """
    Parse output from both HCCN (Ascend) and NVIDIA styles and unify the result.
    Unified output format: { item: {status: bool, output: str}}
    """
    result_dict = {}

    current_item = None
    current_lines = []

    for raw_line in output.splitlines():
        line = raw_line.strip()

        # ========== Ascend ==========
        # Start Detection：xxx
        if line.startswith("Start Detection："):
            current_item = line.replace("Start Detection：", "").strip()
            current_lines = []
            continue

        # [RESULT] xxx: Success/Failed
        if line.startswith("[RESULT]"):
            if current_item:
                status_str = line.split(":", 1)[1].strip()
                passed = status_str.startswith("Success")

                result_dict[current_item] = {
                    "status": passed,
                    "output": "\n".join(current_lines).strip(),
                }

                current_item = None
                current_lines = []
            continue

        # ========== NVIDIA ==========
        # [Detection] xxx
        if line.startswith("[Detection]"):
            current_item = line.replace("[Detection]", "").strip()
            current_lines = []
            continue

        # Pass：xxx
        if line.startswith("Pass："):
            item = line.replace("Pass：", "").strip()

            result_dict[item] = {
                "status": True,
                "output": "\n".join(current_lines).strip(),
            }

            current_item = None
            current_lines = []
            continue

        # Failed：xxx
        if line.startswith("Failed："):
            item = line.replace("Failed：", "").strip()

            result_dict[item] = {
                "status": False,
                "output": "\n".join(current_lines).strip(),
            }

            current_item = None
            current_lines = []
            continue

        if current_item:
            current_lines.append(line)

    return result_dict


def run_hccn_device_status_check():
    """
    Run a series of HCCN (Huawei Cloud Computing Network) device status checks
    on the target node via remote execution. This includes checking link neighbors,
    physical link status, network health, IP configuration, and gateway settings
    for all Ascend AI accelerators.

    The function uses the 'hccn_tool' command-line utility to query device information
    and verifies the health and connectivity of the Ascend cards in the system.

    Returns:
        dict: A dictionary containing the result of each check, parsed from the output.
    """
    HCCN_STATUS_CHECK_SCRIPT = r"""
run_check() {
    cmd="$1"
    desc="$2"
    echo "Start Detection：$desc"
    if eval "$cmd"; then
        echo "[RESULT] $desc: Success"
        return 0
    else
        echo "[RESULT] $desc: Failed (code=$?)" >&2
        return 1
    fi
}

run_check "for i in $(echo \"$ASCEND_RT_VISIBLE_DEVICES\" | tr ',' ' ' | xargs); do hccn_tool -i \$i -lldp -g | grep Ifname; done" "link_neighbor_status"
run_check "for i in $(echo \"$ASCEND_RT_VISIBLE_DEVICES\" | tr ',' ' ' | xargs); do hccn_tool -i \$i -link -g; done" "physical_link_status"
run_check "for i in $(echo \"$ASCEND_RT_VISIBLE_DEVICES\" | tr ',' ' ' | xargs); do hccn_tool -i \$i -net_health -g; done" "network_health_status"
run_check "for i in $(echo \"$ASCEND_RT_VISIBLE_DEVICES\" | tr ',' ' ' | xargs); do hccn_tool -i \$i -ip -g; done" "gpu_ip_config"
"""
    print("\n----------------------------------------\n")
    print(f"[UC] Starting to check the Ascend card status, node IP: {MASTER_IP}")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ASCEND_RT_VISIBLE_DEVICES
    output = run_remote(
        MASTER_IP,
        HCCN_STATUS_CHECK_SCRIPT,
        env={"ASCEND_RT_VISIBLE_DEVICES": ASCEND_RT_VISIBLE_DEVICES},
    )

    # Parsing Output
    result_dict = parse_hccn_output(output)

    overall_status = all(
        v.get("status", False) for v in result_dict.values() if isinstance(v, dict)
    )

    result_dict["device_status_check"] = {"status": overall_status}

    return result_dict


def run_nvidia_device_status_check():
    """
    Execute a comprehensive set of NVIDIA GPU device status checks
    on the target node via remote execution. This includes verifying:
    - Driver loading,
    - GPU device detection,
    - GPU temperature (<85°C),
    - Physical link status (PCIe) for each GPU,
    - Full interconnection topology (using 'nvidia-smi topo -m'),
    - CUDA compiler availability and version.

    The script uses standard NVIDIA tools like 'nvidia-smi', 'lsmod', and 'nvcc'
    to validate the health and configuration of the GPU system.

    Returns:
        dict: A dictionary containing the result of each check, parsed from the output.
    """
    NVIDIA_STATUS_CHECK_SCRIPT = r"""
run_check() {
    cmd="$1"
    desc="$2"
    echo "[Detection] $desc"
    if eval "$cmd"; then
        echo "Pass：$desc"
        return 0
    else
        echo "Failed：$desc (code=$?)" >&2
        return 1
    fi
}

run_check "lsmod | grep -q nvidia" "Check the driver loading"
run_check "nvidia-smi -L | grep -q GPU" "Checking the Device List"
run_check "nvidia-smi -q | awk '/GPU Current Temp/ && \$6!=\"N/A\" {exit \$4>85?1:0}'" "Check temperature (<85°C)"

GPU_COUNT=$(nvidia-smi -L | wc -l)

for i in $(seq 0 $((GPU_COUNT-1))); do
    run_check "nvidia-smi -i $i -q | grep -q 'GPU Link Info'" "Check the physical link information of GPU$i" "WARN"
done

run_check "nvidia-smi topo -m | awk 'BEGIN{all_ok=1} /^GPU/ {for(i=2;i<=NF;i++) if($i==\"-\") all_ok=0} END{exit !all_ok}'" "Check full card interconnection" "WARN"
run_check "nvcc --version | grep -q 'release'" "CUDA version"
"""
    print("\n----------------------------------------\n")
    print(f"[UC] Starting to check the NVIDIA card status, node IP: {MASTER_IP}")
    output = run_remote(MASTER_IP, NVIDIA_STATUS_CHECK_SCRIPT)

    # Parsing Output
    result_dict = parse_hccn_output(output)

    return result_dict


def get_remote_card_ips(ip: str, card_ids: str):
    remote_cmd = f"""
tool="/usr/local/Ascend/driver/tools/hccn_tool"
IFS=',' read -ra card_array <<< "{card_ids}"
declare -a ips
for card_id in "${{card_array[@]}}"; do
    output=$("$tool" -i "$card_id" -ip -g 2>/dev/null)
    ip=$(echo "$output" | grep -oP "ipaddr:\\K[\\d.]+" || echo "")
    [ -n "$ip" ] && ips+=("$ip")
done
printf "%s\\n" "${{ips[@]}}" | sort -V
"""
    try:
        result = subprocess.run(
            f"ssh -i \"{str(LOCAL_SSH_KEY)}\" root@{ip} bash -c '{remote_cmd}'",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip().splitlines()
    except subprocess.CalledProcessError:
        print(f"Failed to get NPU card IPs on remote machine {ip}", file=sys.stderr)
        sys.exit(1)


def remote_hccn_card_ping_test(ip: str, card_ids: str):
    """
    Execute HCCN card intra-node ping test on a remote machine (SSH + execution + structured results)
    :param ip: remote machine IP
    :param card_ids: list of card IDs, e.g., "0,1,2,3"
    :return: result_dict
    """

    remote_cmd = f"""
function intra_node_hccn_card_ping_test() {{
    IFS=',' read -ra card_array <<< "{card_ids}"
    declare -A card_ip_map

    tool="/usr/local/Ascend/driver/tools/hccn_tool"

    echo "[INFO] Retrieving HCCN card IPs..."

    # Step 1: get IP address for each card
    for card_id in "${{card_array[@]}}"; do
        output=$("$tool" -i "$card_id" -ip -g 2>/dev/null)
        ip=$(echo "$output" | grep -oP "ipaddr:\\K[\\d.]+" || echo "")
        if [ -n "$ip" ]; then
            card_ip_map[$card_id]="$ip"
        else
            echo "[RESULT] card_$card_id: Failed to get IP"
        fi
    done

    if [ ${{#card_ip_map[@]}} -eq 0 ]; then
        echo "[FATAL] No HCCN card IPs detected"
        return
    fi

    echo "[INFO] Retrieved card IP mapping:"
    for k in "${{!card_ip_map[@]}}"; do
        echo "  Card $k -> ${{card_ip_map[$k]}}"
    done

    # Step 2: ping test between HCCN cards
    for src_id in "${{!card_ip_map[@]}}"; do
        for dst_id in "${{!card_ip_map[@]}}"; do
            if [ "$src_id" != "$dst_id" ]; then
                pair="local_card_${{src_id}} to local_card_${{dst_id}}"
                dst_ip="${{card_ip_map[$dst_id]}}"

                echo "[TEST] $pair ($dst_ip)"

                if "$tool" -i "$src_id" -ping -g address "$dst_ip" >/dev/null 2>&1; then
                    echo "[RESULT] $pair: Success"
                else
                    echo "[RESULT] $pair: Failed"
                fi
            fi
        done
    done
}}

intra_node_hccn_card_ping_test
"""

    print(f"\n[INFO] Start Detection {ip} HCCN card intra-node Ping...")

    escaped_cmd = shlex.quote(remote_cmd)

    ssh_cmd = [
        "ssh",
        "-q",
        "-o",
        "LogLevel=ERROR",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-i",
        str(LOCAL_SSH_KEY),
        f"root@{ip}",
        "bash",
        "-c",
        escaped_cmd,
    ]

    process = subprocess.Popen(
        ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    output_lines = []
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line.strip())

    process.wait()

    result_dict = {}

    for line in output_lines:
        if line.startswith("[RESULT]"):
            line = line.replace("[RESULT]", "").strip()
            key, status_part = line.split(":", 1)
            status = status_part.strip().startswith("Success")

            result_dict[key.strip()] = {"status": status, "output": ""}

    return result_dict


def remote_local_hccn_cards_ping_test(
    src_type: str, src_ip: str, src_card: str, dst_ip: str
):
    """
    Local or remote HCCN card ping test, return unified result_dict
    :param src_type: "local" or "remote"
    :param src_ip: remote machine IP (if src_type=="remote")
    :param src_card: source card ID
    :param dst_ip: destination IP
    :return: dict -> { "card_X_to_card_Y": {"status": True/False, "output": "..."}, "global_status": {...} }
    """
    HCCN_TOOL = "/usr/local/Ascend/driver/tools/hccn_tool"

    if src_type == "local":
        print(f"[TEST] LOCAL card {src_card} to REMOTE {dst_ip}")
        pair_key = f"local_card_{src_card} to remote_{dst_ip.replace('.', '_')}"

        cmd = (
            f"ssh -q -o LogLevel=ERROR "
            f"-o StrictHostKeyChecking=no "
            f"-o UserKnownHostsFile=/dev/null "
            f'-i "{str(LOCAL_SSH_KEY)}" '
            f"root@{MASTER_IP} {HCCN_TOOL} -i {src_card} -ping -g address {dst_ip}"
        )
    else:
        print(f"[TEST] REMOTE {src_ip} card {src_card} to LOCAL {dst_ip}")
        pair_key = f"remote_card_{src_card} to local_{dst_ip.replace('.', '_')}"

        cmd = (
            f"ssh -q -o LogLevel=ERROR "
            f"-o StrictHostKeyChecking=no "
            f"-o UserKnownHostsFile=/dev/null "
            f'-i "{str(LOCAL_SSH_KEY)}" '
            f"root@{src_ip} {HCCN_TOOL} -i {src_card} -ping -g address {dst_ip}"
        )

    result_dict = {pair_key: {"status": False, "output": ""}}
    try:
        completed = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if completed.returncode == 0:
            print(f"[RESULT] {pair_key}: SUCCESS")
        else:
            print(f"[RESULT] {pair_key}: FAILED")

        result_dict[pair_key]["status"] = completed.returncode == 0

    except Exception as e:
        print(f"[ERROR] Ping test failed: {e}", file=sys.stderr)
        result_dict[pair_key]["status"] = False

    result_dict["global_status"] = {
        "status": result_dict[pair_key]["status"],
        "output": "HCCN local/remote ping global status",
    }

    return result_dict


def run_check_hccn_ping():
    """
    Execute HCCN card ping test including master intra-node ping, local<->remote ping, and remote inter-node ping.
    Returns unified result_dict with global status.
    """
    # ASCEND_RT_VISIBLE_DEVICES = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")

    print("\n----------------------------------------\n")
    print("[UC] start check HCCN ping. Please ensure running at master machine.")

    all_results = {}

    # Get master node card IP list
    local_card_ips = get_remote_card_ips(MASTER_IP, ASCEND_RT_VISIBLE_DEVICES)
    remote_machines = WORKER_IP.split(",") if WORKER_IP else []

    # 1. Master node intra-card ping
    master_result = remote_hccn_card_ping_test(MASTER_IP, ASCEND_RT_VISIBLE_DEVICES)
    all_results.update(master_result)

    # 2. Local <-> Remote node ping
    if NODE_NUM > 1:
        print("\n[INFO] === Local <-> Remote node ping test ===")
        for remote_ip in remote_machines:
            remote_card_ips = get_remote_card_ips(remote_ip, ASCEND_RT_VISIBLE_DEVICES)

            # Local ping remote
            print(f"[INFO] Master node card ping remote node {remote_ip} ...")
            for local_idx, local_ip in enumerate(local_card_ips):
                for remote_card_ip in remote_card_ips:
                    res = remote_local_hccn_cards_ping_test(
                        "local", "", local_idx, remote_card_ip
                    )
                    all_results.update(res)

            # Remote ping local
            print(f"[INFO] Remote node {remote_ip} card ping Master node ...")
            for remote_idx, remote_card_ip in enumerate(remote_card_ips):
                for local_ip_card in local_card_ips:
                    res = remote_local_hccn_cards_ping_test(
                        "remote", remote_ip, remote_idx, local_ip_card
                    )
                    all_results.update(res)

    # 3. Remote inter-node ping
    if len(remote_machines) > 1:
        print("\n[INFO] === Remote inter-node ping test ===")
        for i in range(len(remote_machines)):
            for j in range(i + 1, len(remote_machines)):
                ip1, ip2 = remote_machines[i], remote_machines[j]
                ips1 = get_remote_card_ips(ip1, ASCEND_RT_VISIBLE_DEVICES)
                ips2 = get_remote_card_ips(ip2, ASCEND_RT_VISIBLE_DEVICES)

                # ip1 -> ip2
                print(f"[INFO] Node {ip1} ping Node {ip2} ...")
                for idx1, card1_ip in enumerate(ips1):
                    for card2_ip in ips2:
                        res = remote_local_hccn_cards_ping_test(
                            "remote", ip1, idx1, card2_ip
                        )
                        all_results.update(res)

                # ip2 -> ip1
                print(f"[INFO] Node {ip2} ping Node {ip1} ...")
                for idx2, card2_ip in enumerate(ips2):
                    for card1_ip in ips1:
                        res = remote_local_hccn_cards_ping_test(
                            "remote", ip2, idx2, card1_ip
                        )
                        all_results.update(res)

    # Calculate global status
    global_status = all(
        r["status"] for k, r in all_results.items() if k != "global_status"
    )
    all_results["hccn_ping_check"] = {"status": global_status}

    return all_results


def run_check_nvidia_ping():
    """
    Master <-> Worker ping test in multi-node environment
    :return: result_dict -> { "master->WORKER_IP": {"status": True/False, "output": "..."},
                               "WORKER_IP->master": {...}, "global_status": True/False }
    """
    print("\n----------------------------------------\n")
    print("[UC] start check NVIDIA ping. Please ensure running at master machine.\n")

    remote_machines = WORKER_IP.split(",") if WORKER_IP else []
    result_dict = {}

    if NODE_NUM > 1:
        print(
            "--------------------------NVIDIA multi-node environment: test ping between nodes -----------------------------\n"
        )

        for remote_ip in remote_machines:
            # master -> worker
            pair_key = f"master {MASTER_IP}-> worker {remote_ip}"
            print(f"Testing ping: master {MASTER_IP} --> worker {remote_ip}")
            ping_cmd = f"ping -c 4 -W 1 {remote_ip}"
            completed = subprocess.run(
                ping_cmd, shell=True, capture_output=True, text=True
            )
            output_text = (
                (completed.stdout or "").strip()
                + "\n"
                + (completed.stderr or "").strip()
            )
            status = completed.returncode == 0
            result_dict[pair_key] = {
                "status": status,
                "output": output_text if status else f"Failed\n{output_text}",
            }
            print(f"{pair_key} {'Ping OK' if status else 'Ping Failed'}\n")

            # worker -> master
            pair_key = f"worker {remote_ip}-> master {MASTER_IP}"
            print(f"Testing ping: worker {remote_ip} --> master {MASTER_IP}")
            ssh_ping_cmd = (
                f"ssh -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
                f'-o LogLevel=ERROR -o ConnectTimeout=5 -i "{str(LOCAL_SSH_KEY)}" '
                f'root@{remote_ip} "ping -c 4 -W 1 {MASTER_IP}"'
            )
            completed = subprocess.run(
                ssh_ping_cmd, shell=True, capture_output=True, text=True
            )
            output_text = (
                (completed.stdout or "").strip()
                + "\n"
                + (completed.stderr or "").strip()
            )
            status = completed.returncode == 0
            result_dict[pair_key] = {
                "status": status,
            }
            print(f"{pair_key} {'Ping OK' if status else 'Ping Failed'}\n")

    # Calculate global status
    global_status = (
        all(v["status"] for v in result_dict.values()) if result_dict else True
    )
    result_dict["nvidia_ping_check"] = {"status": global_status}

    return result_dict


def run_hccn_tls_check(MASTER_IP: str, WORKER_IP: str) -> Dict[str, Any]:
    """
    Check whether the TLS switch of all HCCN cards on master and worker nodes is 0.
    Card IDs are read from the ASCEND_RT_VISIBLE_DEVICES environment variable, e.g., "0,1".

    Return format:
    {
        "tls_check": {
            "status": True/False,
            "master_card_0": {"card_id": "0", "status": True},
            "master_card_1": {"card_id": "1", "status": False},
            "worker_card_0": {"card_id": "0", "status": True},
            ...
        }
    }
    """

    global TLS_CHECK_RESULT
    TLS_CHECK_RESULT = True

    print("\n----------------------------------------\n")
    print(f"[UC] Starting to check the HCCN card TLS status")
    devices = ASCEND_RT_VISIBLE_DEVICES
    if not devices:
        raise EnvironmentError(
            "Environment variable ASCEND_RT_VISIBLE_DEVICES is not set"
        )

    card_ids = [c.strip() for c in devices.split(",") if c.strip()]
    if not card_ids:
        raise ValueError(f"Failed to parse ASCEND_RT_VISIBLE_DEVICES: {devices}")

    result: Dict[str, Any] = {"tls_check": {"status": True}}

    def check_node(node_ip: str, node_prefix: str):
        """
        Check TLS switch status on a single node for all HCCN cards.
        """
        global TLS_CHECK_RESULT
        if not node_ip:
            return

        for card_id in card_ids:
            # Construct SSH command
            print(f"[TEST] {node_prefix}_card_{card_id} ({node_ip}) TLS checking...")
            cmd = (
                f"ssh -i {str(LOCAL_SSH_KEY)} -q -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@{node_ip} "
                f"'{shlex.quote(f'/usr/local/Ascend/driver/tools/hccn_tool -i {card_id} -tls -g | head -n 1')}'"
            )

            try:
                output = subprocess.check_output(
                    cmd, shell=True, text=True, stderr=subprocess.DEVNULL
                ).strip()
            except subprocess.CalledProcessError:
                status = False
            else:
                # Extract TLS switch value from the first line
                import re

                match = re.search(r"tls switch\[(\d+)\]", output)
                if match and match.group(1) == "0":
                    status = True
                else:
                    status = False

            key = f"{node_prefix}_card_{card_id}"
            result["tls_check"][key] = {"card_id": card_id, "status": status}

            if status:
                print(f"[RESULT] {node_prefix}_card_{card_id}: PASS (switch=0)")
            else:
                print(f"[RESULT] {node_prefix}_card_{card_id}: FAIL")
                TLS_CHECK_RESULT = False

    # Check master node
    check_node(MASTER_IP, "master")
    # Check worker node
    check_node(WORKER_IP, "worker")

    result["tls_check"]["status"] = TLS_CHECK_RESULT
    return result


MODEL_WEIGHT_FILES_LIST = []


def get_llm_weight_files_list(MODEL_PATH, result_dict):
    global MODEL_WEIGHT_FILES_LIST
    result_dict["get_llm_weight_files_list"] = {"status": False, "output": ""}

    print(f"[UC] Get LLM weight files list: {MODEL_PATH}")

    if not os.path.isdir(MODEL_PATH):
        msg = f"Directory does not exist: {MODEL_PATH}"
        print(msg, file=sys.stderr)
        result_dict["get_llm_weight_files_list"]["output"] = msg
        return result_dict

    os.chdir(MODEL_PATH)

    print("\nSearching for all .safetensors files...")
    MODEL_WEIGHT_FILES_LIST = [
        f for f in os.listdir(".") if f.endswith(".safetensors") and os.path.isfile(f)
    ]

    print(f"Found {len(MODEL_WEIGHT_FILES_LIST)} files")

    if len(MODEL_WEIGHT_FILES_LIST) == 0:
        msg = f"No weight files found in {MODEL_PATH}"
        print(msg, file=sys.stderr)
        result_dict["get_llm_weight_files_list"]["output"] = msg
        return result_dict

    if len(MODEL_WEIGHT_FILES_LIST) > 1:
        # Extract N-of-M pattern from filenames
        total_chunks_from_names = 0
        for f in MODEL_WEIGHT_FILES_LIST:
            match = re.search(r"(\d+)-of-(\d+)", f)
            if match:
                total_chunks_from_names = int(match.group(2).lstrip("0") or 0)
                break

        actual_file_count = len(MODEL_WEIGHT_FILES_LIST)

        if total_chunks_from_names == actual_file_count:
            print(
                f"Validation passed: Total chunks in filenames ({total_chunks_from_names}) matches actual count ({actual_file_count})"
            )
        else:
            msg = (
                f"Validation failed: Total chunks in filenames {total_chunks_from_names}, "
                f"but actually found {actual_file_count} files"
            )
            print(msg, file=sys.stderr)
            result_dict["get_llm_weight_files_list"]["output"] = msg
            return result_dict

    result_dict["get_llm_weight_files_list"]["status"] = True
    return result_dict


def get_weight_files_hash(MODEL_PATH, HF_MODEL_NAME, MIDDLE_PAGE, result_dict):
    result_dict["get_weight_files_hash"] = {"status": False, "output": ""}

    output_path = os.path.join(MODEL_PATH, f"{HF_MODEL_NAME}.sha256")

    if not os.path.exists(output_path):
        print(
            "Weight hash file does not exist, retrieving from HuggingFace. Ensure internet access!"
        )

        for file in MODEL_WEIGHT_FILES_LIST:
            weight_file = file.replace("./", "")
            page = f"https://huggingface.co/{MIDDLE_PAGE}/{HF_MODEL_NAME}/blob/main/{weight_file}"

            max_retry = 3
            retry_delay = 2
            hash_value = ""

            for i in range(1, max_retry + 1):
                try:
                    cmd = (
                        f'curl -fsSL -k --connect-timeout 10 "{page}" | '
                        r'grep -oE "[a-f0-9]{64}" | head -n 1'
                    )
                    result = subprocess.check_output(
                        cmd, shell=True, stderr=subprocess.DEVNULL
                    )
                    hash_value = result.decode().strip()

                    if re.fullmatch(r"[a-f0-9]{64}", hash_value):
                        break

                    if i < max_retry:
                        print(f"Request failed, retry {i}...", file=sys.stderr)
                        time.sleep(retry_delay)

                except subprocess.CalledProcessError:
                    pass

            if not re.fullmatch(r"[a-f0-9]{64}", hash_value):
                msg = f"Cannot get valid hash: {weight_file}"
                print(msg, file=sys.stderr)
                result_dict["get_weight_files_hash"]["output"] = msg
                return result_dict

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(f"{hash_value}  {file}\n")

    else:
        print(f"{output_path} already exists")

    result_dict["get_weight_files_hash"]["status"] = True
    return result_dict


def run_check_model_weight():
    result_dict = {}

    print("\n----------------------------------------\n")
    print(f"[UC] Check model weight: {MODEL_PATH}")

    # Step 1: Get model files list
    result_dict = get_llm_weight_files_list(MODEL_PATH, result_dict)

    # Step 2: Get HuggingFace hash
    if result_dict.get("get_llm_weight_files_list", {}).get("status"):
        result_dict = get_weight_files_hash(
            MODEL_PATH, HF_MODEL_NAME, MIDDLE_PAGE, result_dict
        )
    else:
        result_dict["global_status"] = {"status": False}
        return result_dict

    # Step 3: Verify sha256
    result_dict["sha256_verification"] = {"status": False, "output": ""}
    os.chdir(MODEL_PATH)
    sha256_files = [f for f in os.listdir(".") if f.endswith(".sha256")]

    if len(sha256_files) == 0:
        msg = "Error: No .sha256 files found"
        print(msg, file=sys.stderr)
        result_dict["sha256_verification"]["output"] = msg
    elif len(sha256_files) > 1:
        msg = f"Error: Multiple .sha256 files found: {sha256_files}"
        print(msg, file=sys.stderr)
        result_dict["sha256_verification"]["output"] = msg
    else:
        sha256_file = sha256_files[0]
        try:
            subprocess.check_call(f"sha256sum -c {sha256_file}", shell=True)
            print("All .safetensors files verified successfully")
            result_dict["sha256_verification"]["status"] = True
        except subprocess.CalledProcessError:
            msg = "Model weight verification failed, exiting"
            print(msg, file=sys.stderr)
            result_dict["sha256_verification"]["output"] = msg

    # Calculate global status
    global_status = all(v.get("status", False) for k, v in result_dict.items())
    result_dict["model_weight_check"] = {"status": global_status}

    return result_dict


def create_worker(device_id: int) -> UcmKVStoreBaseV1:
    module_path = "ucm.store.pipeline.connector"
    class_name = "UcmPipelineStore"
    config = {}
    config["store_pipeline"] = "Posix"
    config["posix_io_engine"] = "aio"
    config["storage_backends"] = STORAGE_BACKENDS
    config["tensor_size"] = SHARD_SIZE
    config["shard_size"] = SHARD_SIZE
    config["device_id"] = device_id
    return UcmConnectorFactoryV1.create_connector(class_name, config, module_path)


def make_array(size, alignment=262144, dtype=np.uint8) -> tuple[np.ndarray, mmap.mmap]:
    itemsize = np.dtype(dtype).itemsize
    total_bytes = size * itemsize
    mm = mmap.mmap(-1, total_bytes + alignment)
    raw_array = np.frombuffer(mm, dtype=np.uint8, count=total_bytes + alignment)
    raw_ptr = raw_array.__array_interface__["data"][0]
    aligned_addr = (raw_ptr + alignment - 1) & ~(alignment - 1)
    offset = aligned_addr - raw_ptr
    array = raw_array[offset : offset + total_bytes].view(dtype=dtype)
    return array, mm


def dump(epoch, device_id, worker, block_ids, block_ptr, dump_bw_list):
    total_size = SHARD_SIZE * SHARD_NUMBER * BLOCK_NUMBER
    costs = []
    for i in range(SHARD_NUMBER):
        idxes = [i for _ in range(BLOCK_NUMBER)]
        ptrs = [[ptr + i * SHARD_SIZE] for ptr in block_ptr]
        tp = time.perf_counter()
        task = worker.dump_data(block_ids, idxes, ptrs)
        worker.wait(task)
        costs.append(time.perf_counter() - tp)
    total_cost = np.sum(costs)
    bw = total_size / total_cost / 1e9

    dump_bw_list.append(bw)

    print(
        f"epoch={epoch:03}, worker={device_id:02}, "
        f"dump=[{SHARD_SIZE} x {BLOCK_NUMBER} x {SHARD_NUMBER}], "
        f"avg_cost={np.average(costs) * 1e3:.3f}ms, "
        f"p99_cost={np.percentile(costs, 99) * 1e3:.3f}ms, "
        f"total_cost={total_cost * 1e3:.3f}ms, "
        f"bw={bw:.3f}GB/s."
    )


def load(epoch, device_id, worker, block_ids, block_ptr, load_bw_list):
    total_size = SHARD_SIZE * SHARD_NUMBER * BLOCK_NUMBER
    costs = []
    for i in range(SHARD_NUMBER):
        idxes = [i for _ in range(BLOCK_NUMBER)]
        ptrs = [[ptr + i * SHARD_SIZE] for ptr in block_ptr]
        tp = time.perf_counter()
        task = worker.load_data(block_ids, idxes, ptrs)
        worker.wait(task)
        costs.append(time.perf_counter() - tp)
    total_cost = np.sum(costs)
    bw = total_size / total_cost / 1e9

    load_bw_list.append(bw)

    print(
        f"epoch={epoch:03}, worker={device_id:02}, "
        f"load=[{SHARD_SIZE} x {BLOCK_NUMBER} x {SHARD_NUMBER}], "
        f"avg_cost={np.average(costs) * 1e3:.3f}ms, "
        f"p99_cost={np.percentile(costs, 99) * 1e3:.3f}ms, "
        f"total_cost={total_cost * 1e3:.3f}ms, "
        f"bw={bw:.3f}GB/s."
    )


def worker_loop(device_id, barrier, dump_bw_list, load_bw_list):
    store = create_worker(device_id)
    block_ids = [secrets.token_bytes(16) for _ in range(BLOCK_NUMBER)]

    mmap_handles = []
    block_data = []
    for _ in range(BLOCK_NUMBER):
        arr, mm = make_array(SHARD_SIZE * SHARD_NUMBER)
        block_data.append(arr)
        mmap_handles.append(mm)

    block_ptr = [block.ctypes.data for block in block_data]
    barrier.wait()

    for epoch in range(DUMP_EPOCH_NUMBER):
        dump(epoch, device_id, store, block_ids, block_ptr, dump_bw_list)
        barrier.wait()

    for epoch in range(LOAD_EPOCH_NUMBER):
        load(epoch, device_id, store, block_ids, block_ptr, load_bw_list)
        barrier.wait()

    for mm in mmap_handles:
        try:
            mm.close()
        except Exception:
            pass


def run_bandwidth_check():
    manager = multiprocessing.Manager()
    dump_bw_list = manager.list()
    load_bw_list = manager.list()

    barrier = multiprocessing.Barrier(WORKER_NUMBER)
    workers = []

    for i in range(WORKER_NUMBER):
        p = multiprocessing.Process(
            target=worker_loop, args=(i, barrier, dump_bw_list, load_bw_list)
        )
        workers.append(p)
        p.start()

    for w in workers:
        w.join()

    avg_dump = np.mean(dump_bw_list) if len(dump_bw_list) > 0 else 0.0
    avg_load = np.mean(load_bw_list) if len(load_bw_list) > 0 else 0.0

    print(f"\n==== FINAL AVERAGE BANDWIDTH ====")
    print(f" Dump BW (avg): {avg_dump:.3f} GB/s")
    print(f" Load BW (avg): {avg_load:.3f} GB/s")

    return {"dump": round(avg_dump, 3), "load": round(avg_load, 3)}
