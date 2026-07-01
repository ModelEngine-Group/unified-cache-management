# UCM patch for vllm-ascend 0.21.0:
# Remove bind_memory call from bind_threads to avoid migratepages failure
import psutil
from vllm_ascend.cpu_binding import CpuAlloc, execute_command


def bind_threads(self) -> None:
    thread_message, _ = execute_command(["ps", "-Te"])
    threads_map = CpuAlloc.get_threads_map(thread_message)
    main_pid = str(psutil.Process().pid)
    current_npu = self.device_info.running_npu_list[self.rank_id]
    self.bind(main_pid, self.assign_main[current_npu], True)
    for acl_thread in threads_map.get(main_pid, {}).get("acl_thread", []):
        self.bind(acl_thread, self.assign_acl[current_npu], False)
    for release_thread in threads_map.get(main_pid, {}).get("release_thread", []):
        self.bind(release_thread, self.assign_rel[current_npu], False)
