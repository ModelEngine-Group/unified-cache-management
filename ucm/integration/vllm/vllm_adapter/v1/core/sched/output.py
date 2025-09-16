from __future__ import annotations

from dataclasses import dataclass

import vllm.v1.core.sched.output as vllm_v1_scheduler_output


@dataclass
class SchedulerOutput(vllm_v1_scheduler_output.SchedulerOutput):
    # modified slots by sparse algorithm
    req_sparsed_slots: dict[str, int] = None


vllm_v1_scheduler_output.SchedulerOutput = SchedulerOutput
