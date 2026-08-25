#!/usr/bin/env python3
import os
import re
import sys

TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
UNIFIEDCACHE_VOLUME_NAMES = []


def env(name):
    value = os.environ.get(name, "").strip()
    if value:
        return value
    raise RuntimeError(f"missing required env: {name}")


def get(session, url, headers):
    response = session.get(url, headers=headers, verify=CA_PATH, timeout=10)
    response.raise_for_status()
    return response.json()


def choose_volumes(volumes):
    allowed = set(UNIFIEDCACHE_VOLUME_NAMES)
    pvc_volumes_by_name = {}

    for v in volumes:
        volume_name = v.get("name")
        name_ok = volume_name in allowed
        pvc = v.get("persistentVolumeClaim") or {}
        has_claim = pvc.get("claimName")

        if name_ok and has_claim:
            pvc_volumes_by_name[volume_name] = v
    if not pvc_volumes_by_name:
        raise RuntimeError("pod has no PVC-backed volumes from unifiedcacheStorage")

    ordered_volumes = []
    missing_names = []
    for volume_name in UNIFIEDCACHE_VOLUME_NAMES:
        volume = pvc_volumes_by_name.get(volume_name)
        if volume is None:
            missing_names.append(volume_name)
            continue
        ordered_volumes.append(volume)

    if missing_names:
        raise RuntimeError(
            "some unifiedcacheStorage volumes are not PVC-backed or missing from pod spec: "
            + ", ".join(missing_names)
        )
    return ordered_volumes


def patch_kvcs_store_ids(content, kvcache_store_id):
    updated, count = re.subn(
        r"(?m)^(\s*kvcs_store_id:\s*).*$",
        lambda match: f'{match.group(1)}"{kvcache_store_id}"',
        content,
    )
    if count == 0:
        raise RuntimeError("no kvcs_store_id entries found in runtime UCM config")
    return updated


def main():
    import requests

    api_base = (
        f"https://{env('KUBERNETES_SERVICE_HOST')}:{env('KUBERNETES_SERVICE_PORT')}"
    )
    pod_name = env("POD_NAME")
    pod_namespace = env("POD_NAMESPACE")
    runtime_ucm_config_path = os.environ.get("UCM_CONFIG_FILE", "").strip()
    if not runtime_ucm_config_path:
        raise RuntimeError("missing UCM_CONFIG_FILE")
    if not os.path.exists(runtime_ucm_config_path):
        raise RuntimeError(f"runtime UCM config not found: {runtime_ucm_config_path}")

    with open(TOKEN_PATH, "r", encoding="utf-8") as token_file:
        token = token_file.read().strip()

    headers = {"Authorization": f"Bearer {token}"}
    session = requests.Session()

    pod = get(
        session,
        f"{api_base}/api/v1/namespaces/{pod_namespace}/pods/{pod_name}",
        headers,
    )
    selected_volumes = choose_volumes(pod.get("spec", {}).get("volumes") or [])
    kvcache_store_ids = []
    selected_volume_names = []

    for selected_volume in selected_volumes:
        pvc_name = selected_volume["persistentVolumeClaim"]["claimName"]
        pvc = get(
            session,
            f"{api_base}/api/v1/namespaces/{pod_namespace}/persistentvolumeclaims/{pvc_name}",
            headers,
        )
        pv_name = (pvc.get("spec") or {}).get("volumeName")
        if not pv_name:
            raise RuntimeError(f"PVC {pvc_name!r} is not yet bound to a PV")

        pv = get(session, f"{api_base}/api/v1/persistentvolumes/{pv_name}", headers)
        kvcache_store_id = (
            ((pv.get("spec") or {}).get("csi") or {}).get("volumeAttributes") or {}
        ).get("kvcacheStoreId")
        if not kvcache_store_id:
            raise RuntimeError(
                f"PV {pv_name!r} missing csi.volumeAttributes.kvcacheStoreId"
            )
        kvcache_store_ids.append(kvcache_store_id)
        selected_volume_names.append(selected_volume["name"])

    patched_store_ids = ":".join(kvcache_store_ids)

    with open(runtime_ucm_config_path, "r", encoding="utf-8") as config_file:
        content = config_file.read()
    updated = patch_kvcs_store_ids(content, patched_store_ids)
    with open(runtime_ucm_config_path, "w", encoding="utf-8") as config_file:
        config_file.write(updated)

    print(
        f"[INFO] patched kvcs_store_id in {runtime_ucm_config_path} "
        f"with kvcacheStoreId={patched_store_ids} from PVC volumes {', '.join(selected_volume_names)}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[WARN] patch_ucm_kvcs_store_id failed: {exc}", file=sys.stderr)
        sys.exit(1)
