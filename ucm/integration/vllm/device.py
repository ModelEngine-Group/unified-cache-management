# -*- coding: utf-8 -*-
"""
Event-based sync between Python compute stream and C++ cache stream.

When dump_data is called, the cache's C++ stream does D2H from device memory.
We must ensure the Python compute stream has finished writing KVCache before
the cache reads. Event sync: record event on compute stream, pass to C++,
cache stream waits for event before D2H. This avoids blocking the CPU.
"""
import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import accumulate
from typing import Dict, List, Optional, Tuple, Union

import torch
from vllm.platforms import current_platform

from ucm.logger import init_logger

logger = init_logger(__name__)


class Device(ABC):
    def __init__(self):
        self.events = []

    @abstractmethod
    def get_event_handle(self) -> int:
        """Return event handle for stream sync. 0 means no event (use synchronize instead)."""
        pass

    @abstractmethod
    def synchronize(self):
        pass

    @abstractmethod
    def destroy_event_handles(self):
        pass

    @abstractmethod
    def get_cpu_affinity(self, local_rank: int) -> Optional[str]:
        """
        Return CPU affinity as a cpulist string, e.g. "0-43,88-131".
        """
        pass

    def split_cores(self, local_rank: int) -> Tuple[List[int], List[int]]:
        """
        Shared split logic for both CUDA and NPU.
        Split each cpulist segment evenly and keep at least one core for worker.
        """
        worker_cores, store_cores = [], []
        cpu_affinity = self.get_cpu_affinity(local_rank)

        if not cpu_affinity:
            return worker_cores, store_cores

        try:
            for part in cpu_affinity.split(","):
                part = part.strip()
                if not part:
                    continue

                if "-" in part:
                    a, b = map(int, part.split("-", 1))
                    if a > b:
                        a, b = b, a
                    seg = list(range(a, b + 1))
                else:
                    seg = [int(part)]

                mid = max(1, len(seg) // 2)
                worker_cores.extend(seg[:mid])
                store_cores.extend(seg[mid:])

            if not worker_cores:
                cores = sorted(os.sched_getaffinity(0))
                if cores:
                    worker_cores = [cores[0]]
                    store_cores = cores[1:]

        except Exception as e:
            logger.error(f"split cores failed, cpu_affinity={cpu_affinity}: {e}")
            return [], []

        logger.info(
            f"[CPU Affinity] rank={local_rank}, cpu_affinity={cpu_affinity}\n"
            f"[worker_cores]={worker_cores}\n"
            f"[store_cores]={store_cores}"
        )
        return worker_cores, store_cores


class CudaDevice(Device):
    def __init__(self):
        super().__init__()

    def get_event_handle(self) -> int:
        try:
            cuda_event = torch.cuda.Event(enable_timing=False)
            stream = torch.cuda.current_stream()
            cuda_event.record(stream)
            handle = int(cuda_event.cuda_event)
            if handle is None or handle == 0:
                return 0
            self.events.append(cuda_event)
            return handle
        except Exception as e:
            logger.error(f"get cuda event handle failed. {e}")
            return 0

    def synchronize(self):
        torch.cuda.current_stream().synchronize()

    def destroy_event_handles(self):
        self.events.clear()

    def get_cpu_affinity(self, local_rank: int) -> Optional[str]:
        """
        CUDA path:
        1. GPU -> PCI -> NUMA -> cpulist
        2. fallback: split current allowed CPUs by local_rank
        """
        try:
            prop = torch.cuda.get_device_properties(local_rank)
            pci_bus_id = (
                f"{prop.pci_domain_id:04x}:"
                f"{prop.pci_bus_id:02x}:"
                f"{prop.pci_device_id:02x}.0"
            )

            numa_path = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
            if os.path.exists(numa_path):
                with open(numa_path) as f:
                    numa_node = int(f.read().strip())

                if numa_node >= 0:
                    cpu_list_path = f"/sys/devices/system/node/node{numa_node}/cpulist"
                    if os.path.exists(cpu_list_path):
                        with open(cpu_list_path) as f:
                            return f.read().strip()
        except Exception as e:
            logger.warning(f"get cuda cpu affinity from numa failed: {e}")

        try:
            cores = sorted(os.sched_getaffinity(0))
            if not cores:
                return None

            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            total_devices = (
                len([x.strip() for x in visible.split(",") if x.strip()])
                if visible
                else torch.cuda.device_count()
            )

            if total_devices <= 0 or local_rank < 0 or local_rank >= total_devices:
                logger.warning(
                    f"[CPU Affinity] invalid cuda fallback split: "
                    f"local_rank={local_rank}, total_devices={total_devices}"
                )
                return None

            base = len(cores) // total_devices
            extra = len(cores) % total_devices
            start = local_rank * base + min(local_rank, extra)
            length = base + (1 if local_rank < extra else 0)
            sliced = cores[start : start + length]

            if not sliced:
                return None

            parts = []
            s = e = sliced[0]
            for c in sliced[1:]:
                if c == e + 1:
                    e = c
                else:
                    parts.append(f"{s}-{e}" if s != e else str(s))
                    s = e = c
            parts.append(f"{s}-{e}" if s != e else str(s))

            cpu_affinity = ",".join(parts)
            logger.warning(
                f"[CPU Affinity] fallback to sliced allowed CPUs for cuda rank={local_rank}: "
                f"{cpu_affinity}"
            )
            return cpu_affinity

        except Exception as e:
            logger.error(f"get cuda cpu affinity fallback failed: {e}")
            return None


class NpuDevice(Device):
    @dataclass
    class NpuDeviceInfo:
        npu_id: int
        chip_id: int
        chip_logic_id: Union[int, str]
        chip_name: str
        pcie_info: Optional[str] = None
        numa_id: Optional[int] = None

        @classmethod
        def from_info_line(cls, line: str) -> "NpuDevice.NpuDeviceInfo":
            npu_id, chip_id, chip_logic_id, chip_name = line.strip().split(None, 3)
            chip_logic_id = (
                int(chip_logic_id) if chip_logic_id.isnumeric() else chip_logic_id
            )
            return cls(
                npu_id=int(npu_id),
                chip_id=int(chip_id),
                chip_logic_id=chip_logic_id,
                chip_name=chip_name,
            )

    def __init__(self):
        super().__init__()

    def get_event_handle(self) -> int:
        import acl
        import torch_npu

        try:
            stream = torch_npu.npu.current_stream().npu_stream
            event, ret = acl.rt.create_event()
            if ret != 0:
                logger.error(f"acl create_event failed: {ret}")
                return 0
            self.events.append(event)
            ret = acl.rt.record_event(event, stream)
            if ret != 0:
                logger.error(f"acl record_event failed: {ret}")
                return 0
            handle = int(event)
            if not handle:
                return 0
            return handle
        except Exception as e:
            logger.error(f"get npu event handle failed. {e}")
            return 0

    def synchronize(self):
        torch.npu.current_stream().synchronize()

    def destroy_event_handles(self):
        import acl

        for event in self.events:
            try:
                acl.rt.destroy_event(event)
            except Exception as e:
                logger.error(f"destroy npu event failed. {e}")
        self.events.clear()

    def _execute_command(self, cmd_list: List[str]) -> str:
        try:
            with subprocess.Popen(
                cmd_list,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as p:
                out, err = p.communicate(timeout=1000)

            if p.returncode != 0:
                raise RuntimeError(
                    f"command failed: {cmd_list}, returncode={p.returncode}, "
                    f"stderr={err.decode(errors='ignore')}"
                )
            return out.decode(errors="ignore")
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            raise RuntimeError(f"command timeout: {cmd_list}")

    def _get_visible_device_list_from_env(self) -> Optional[List[int]]:
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get(
            "ASCEND_VISIBLE_DEVICES"
        )
        if not visible:
            return None
        return [int(x.strip()) for x in visible.split(",") if x.strip()]

    def _get_device_id(self, local_rank: int) -> int:
        dev_list = self._get_visible_device_list_from_env()
        if not dev_list:
            return local_rank

        if local_rank < len(dev_list):
            return dev_list[local_rank]

        logger.warning(
            f"[CPU Affinity] local_rank={local_rank} is out of visible NPU range: "
            f"{dev_list}, fallback to local_rank itself."
        )
        return local_rank

    def _get_visible_devices(self) -> List[int]:
        visible = self._get_visible_device_list_from_env()
        if visible:
            return visible

        try:
            return sorted(list(self._get_device_map_info().keys()))
        except Exception:
            return list(range(torch.npu.device_count()))

    def _get_device_map_info(self) -> Dict[int, "NpuDevice.NpuDeviceInfo"]:
        device_map_info: Dict[int, NpuDevice.NpuDeviceInfo] = {}
        device_map = (
            self._execute_command(["npu-smi", "info", "-m"]).strip().split("\n")[1:]
        )

        for line in device_map:
            line = line.strip()
            if not line:
                continue
            try:
                topo = self.NpuDeviceInfo.from_info_line(line)
                if isinstance(topo.chip_logic_id, int):
                    device_map_info[topo.chip_logic_id] = topo
            except (ValueError, IndexError):
                continue

        return device_map_info

    def _get_pcie_info(
        self, device_map_info: Dict[int, "NpuDevice.NpuDeviceInfo"]
    ) -> None:
        keyword = "PCIeBusInfo"

        for device_id, topo in device_map_info.items():
            topo.pcie_info = None

            try:
                pcie_info = (
                    self._execute_command(
                        [
                            "npu-smi",
                            "info",
                            "-t",
                            "board",
                            "-i",
                            f"{topo.npu_id}",
                            "-c",
                            f"{topo.chip_id}",
                        ]
                    )
                    .strip()
                    .split("\n")
                )

                for line_raw in pcie_info:
                    # Normalize spaces to handle variants like:
                    # "PCIe Bus Info : 0000:C1:00.0"
                    # vs "PCIeBusInfo:0000:C1:00.0"
                    line = "".join(line_raw.split())

                    if line.startswith(keyword):
                        topo.pcie_info = line[len(keyword) + 1 :].upper()
                        break

                if topo.pcie_info is None:
                    logger.warning(
                        f"[CPU Affinity] cannot find {keyword} for NPU device {device_id} "
                        f"(npu_id={topo.npu_id}, chip_id={topo.chip_id})"
                    )

            except Exception as e:
                logger.warning(
                    f"[CPU Affinity] failed to get PCIe info for NPU device {device_id} "
                    f"(npu_id={topo.npu_id}, chip_id={topo.chip_id}): {e}"
                )

    def _get_numa_info(
        self, device_map_info: Dict[int, "NpuDevice.NpuDeviceInfo"]
    ) -> None:
        for device_id, topo in device_map_info.items():
            if not topo.pcie_info:
                continue

            pcie_bdf = topo.pcie_info.lower()
            numa_path = f"/sys/bus/pci/devices/{pcie_bdf}/numa_node"

            # 1. sysfs
            try:
                if os.path.exists(numa_path):
                    with open(numa_path) as f:
                        numa_id = int(f.read().strip())
                    if numa_id >= 0:
                        topo.numa_id = numa_id
                        continue
            except Exception as e:
                logger.warning(
                    f"[NUMA] failed to read sysfs NUMA node for device {device_id}, "
                    f"PCI {topo.pcie_info}: {e}"
                )

            # 2. lspci fallback
            try:
                out = self._execute_command(
                    ["/usr/bin/lspci", "-s", topo.pcie_info, "-vvv"]
                )
                m = re.search(r"NUMA\s*node\s*:\s*(\d+)", out, re.IGNORECASE)
                if m:
                    topo.numa_id = int(m.group(1))
            except Exception:
                pass

    def _get_numa_info_v2(
        self, device_map_info: Dict[int, "NpuDevice.NpuDeviceInfo"], devices: List[int]
    ) -> None:
        """
        Fallback when real NPU->NUMA mapping is unavailable:
        distribute visible devices evenly across NUMA nodes.
        """
        numa_nodes = 1
        numa_info = self._execute_command(["lscpu"]).split("\n")
        for line_raw in numa_info:
            line = "".join(line_raw.split())
            if "NUMAnode(s)" not in line:
                continue
            try:
                numa_nodes = int(line.split(":")[-1])
            except Exception:
                numa_nodes = 1
            break

        device_per_numa, tail_device = divmod(len(devices), numa_nodes)
        device_count_per_numa_list = [
            device_per_numa + (i < tail_device) for i in range(numa_nodes)
        ]

        ends = list(accumulate(device_count_per_numa_list))
        starts = [0] + ends[:-1]

        for numa_id, (start, end) in enumerate(zip(starts, ends)):
            for device_id in devices[start:end]:
                if device_id in device_map_info:
                    device_map_info[device_id].numa_id = numa_id

    def _get_cpu_info(self, numa_ids: List[int]) -> Dict[int, List[int]]:
        cpu_idx_tbl: Dict[int, List[int]] = {}
        cpu_info = self._execute_command(["lscpu"]).split("\n")

        for line_raw in cpu_info:
            m = re.match(r"NUMA node(\d+) CPU\(s\):\s*(.*)", line_raw)
            if not m:
                continue

            numa_id = int(m.group(1))
            if numa_id not in numa_ids:
                continue

            cpu_id_ranges = m.group(2).split(",")
            ranges: List[int] = []
            for range_str in cpu_id_ranges:
                range_str = range_str.strip()
                if not range_str:
                    continue
                if "-" in range_str:
                    start, end = map(int, range_str.split("-", 1))
                    if start > end:
                        start, end = end, start
                    ranges.extend(range(start, end + 1))
                else:
                    ranges.append(int(range_str))

            cpu_idx_tbl[numa_id] = ranges

        return cpu_idx_tbl

    def _to_cpulist_str(self, cores: List[int]) -> Optional[str]:
        if not cores:
            return None

        cores = sorted(set(cores))
        parts = []
        s = e = cores[0]

        for c in cores[1:]:
            if c == e + 1:
                e = c
            else:
                parts.append(f"{s}-{e}" if s != e else str(s))
                s = e = c
        parts.append(f"{s}-{e}" if s != e else str(s))
        return ",".join(parts)

    def _fallback_cpu_affinity(self, local_rank: int) -> Optional[str]:
        try:
            cores = sorted(os.sched_getaffinity(0))
            if not cores:
                return None

            visible = self._get_visible_device_list_from_env()
            total_devices = len(visible) if visible else torch.npu.device_count()

            if total_devices <= 0 or local_rank < 0 or local_rank >= total_devices:
                logger.warning(
                    f"[CPU Affinity] invalid npu fallback split: "
                    f"local_rank={local_rank}, total_devices={total_devices}"
                )
                return None

            base = len(cores) // total_devices
            extra = len(cores) % total_devices
            start = local_rank * base + min(local_rank, extra)
            length = base + (1 if local_rank < extra else 0)
            sliced = cores[start : start + length]

            if not sliced:
                return None

            cpu_affinity = self._to_cpulist_str(sliced)
            logger.warning(
                f"[CPU Affinity] fallback to sliced allowed CPUs for npu rank={local_rank}: "
                f"{cpu_affinity}"
            )
            return cpu_affinity

        except Exception as e:
            logger.error(f"get npu cpu affinity fallback failed: {e}")
            return None

    def _get_node_socket_map(self) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """
        Parse `lscpu -e=cpu,node,socket` and return:
          - node_to_socket: {numa_node: socket_id}
          - socket_to_nodes: {socket_id: [numa_node0, numa_node1, ...]}
        """
        node_to_socket: Dict[int, int] = {}
        socket_to_nodes: Dict[int, set] = {}

        out = self._execute_command(["lscpu", "-e=cpu,node,socket"]).splitlines()
        for line in out:
            parts = line.split()
            if len(parts) != 3:
                continue
            if parts[0].upper() == "CPU":
                continue

            _, node_str, socket_str = parts
            if not (node_str.isdigit() and socket_str.isdigit()):
                continue

            node = int(node_str)
            socket = int(socket_str)
            node_to_socket[node] = socket
            socket_to_nodes.setdefault(socket, set()).add(node)

        socket_to_nodes_sorted: Dict[int, List[int]] = {
            s: sorted(nodes) for s, nodes in socket_to_nodes.items()
        }
        return node_to_socket, socket_to_nodes_sorted

    def _merge_cpulist_parts(self, cpulist_parts: List[str]) -> Optional[str]:
        """
        Join multiple cpulist parts without merging adjacent ranges.
        Example:
          ["64-94", "95-127"] -> "64-94,95-127"
        """
        parts = [p.strip() for p in cpulist_parts if p and p.strip()]
        return ",".join(parts) if parts else None

    def get_cpu_affinity(self, local_rank: int) -> Optional[str]:
        """
        NPU path:
        1. NPU -> PCIe -> NUMA
        2. NUMA -> socket
        3. socket -> all NUMA nodes on the same socket
        4. all local NUMA nodes -> merged cpulist

        Return CPU affinity as all CPUs on the device-local socket.
        """
        device_id = self._get_device_id(local_rank)

        try:
            devices = self._get_visible_devices()
            device_map_info = self._get_device_map_info()
            if not device_map_info:
                logger.warning(
                    "[CPU Affinity] empty NPU device info map, fallback to sliced allowed CPUs."
                )
                return self._fallback_cpu_affinity(local_rank)
            # NPU -> PCIe
            self._get_pcie_info(device_map_info)
            # PCIe -> NUMA
            self._get_numa_info(device_map_info)

            if not any(
                device_map_info[d].numa_id is not None
                for d in devices
                if d in device_map_info
            ):
                logger.warning(
                    "[CPU Affinity] failed to get real NPU->NUMA mapping, "
                    "fallback to evenly distributed NUMA mapping."
                )
                self._get_numa_info_v2(device_map_info, devices)

            topo = device_map_info.get(device_id)
            if topo is None or topo.numa_id is None:
                logger.warning(
                    f"[CPU Affinity] cannot find NUMA node for NPU device {device_id}"
                )
                return self._fallback_cpu_affinity(local_rank)

            numa_id = topo.numa_id

            # NUMA -> socket -> all NUMA nodes on the same socket
            node_to_socket, socket_to_nodes = self._get_node_socket_map()
            if numa_id not in node_to_socket:
                logger.warning(
                    f"[CPU Affinity] cannot map NUMA node {numa_id} to socket"
                )
                return self._fallback_cpu_affinity(local_rank)

            socket = node_to_socket[numa_id]
            local_numa_ids = sorted(socket_to_nodes.get(socket, [numa_id]))

            # Collect all CPUs on the local socket
            cpu_idx_tbl = self._get_cpu_info(local_numa_ids)
            cpulist_parts: List[str] = []
            for nid in local_numa_ids:
                nid_cores = cpu_idx_tbl.get(nid, [])
                if not nid_cores:
                    continue
                nid_cpulist = self._to_cpulist_str(nid_cores)
                if nid_cpulist:
                    cpulist_parts.append(nid_cpulist)

            if not cpulist_parts:
                logger.warning(
                    f"[CPU Affinity] cannot find CPU list for local socket NUMA nodes "
                    f"{local_numa_ids}"
                )
                return self._fallback_cpu_affinity(local_rank)

            cpu_affinity = self._merge_cpulist_parts(cpulist_parts)
            logger.info(
                f"[CPU Affinity] NPU device={device_id}, "
                f"pcie_info={topo.pcie_info}, numa_id={numa_id}, "
                f"socket={socket}, local_numa_ids={local_numa_ids}, "
                f"cpu_affinity={cpu_affinity}"
            )
            return cpu_affinity

        except Exception as e:
            logger.warning(f"get npu cpu affinity from numa failed: {e}")
            return self._fallback_cpu_affinity(local_rank)


def create_device() -> Optional[Device]:
    if current_platform.is_cuda_alike():
        return CudaDevice()

    if current_platform.device_type == "npu":
        return NpuDevice()

    return None
