---
license: cc-by-4.0
tags:
- code
- synthetic
- tools
- agents
- software
size_categories:
- 10K<n<100K
---
# Dataset Summary

**SWE-rebench-OpenHands-Trajectories** is a dataset of multi-turn agent trajectories for software engineering tasks, collected
using [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) with OpenHands (v0.54.0) agent scaffolding.
This dataset captures complete agent execution traces as they attempt to resolve real GitHub issues from
[nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench).
Each trajectory contains the agent's step-by-step reasoning, actions, and environmental observations.

| Metric | | [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | [Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k) | [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | [SWE-Gym/OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories) | [R2E-Gym/R2EGym-SFT-Trajectories](https://huggingface.co/datasets/R2E-Gym/R2EGym-SFT-Trajectories) | Ours |
|--------|------|------|------|------|------|------|------|
| **Scaffolding** | | SWE-agent | mini-swe-agent-plus | Closed-source | OpenHands | OpenHands | OpenHands (v0.54.0) |
| **Bootstrapping Model** | Name | claude-3-7-sonnet-20250219<br/>claude-3-5-sonnet-20241022<br/>gpt-4o-2024-08-06 | *unknown\** | Qwen2.5-72B-Instruct<br/>Llama3-70B-Instruct | gpt-4o-2024-08-06<br/>claude-3-5-sonnet-20241022 | Claude-Sonnet-3.5-v2 | Qwen3-Coder-480B-A35B-Instruct |
| | Uses function calling | ✅  | ❌ | ❌ | ✅  | ✅  | ✅ |
| **Repositories** | | 129 | 123 | 1,202 | 11 | *unknown\** | **1,823** |
| **Issues** | Resolved Count | 7,270 | **10,894** | 838 | 294 | 2,048 | 3,792 |
| | Real-world/Synthetic | <span style="color: red;">Synthetic</span> | <span style="color: red;">Synthetic</span> | <span style="color: green;">Real-world</span> | <span style="color: green;">Real-world</span> | <span style="color: green;">Real-world</span> | <span style="color: green;">Real-world</span> |
| **Trajectories** | Total Count | 49,897 | 65,994 | **80,036** | 6,055 | 3,231 | 67,074 |
| | Successful Count | 21,513 | **65,994** | 13,389 | 491 | 3,231 | 32,161 |
| **Turns** | Max Count | 151 | 157 | **408** | 50 | 42 | 100 |
| | Average Count | 30.2 | 34.3 | 26.4 | 18.9 | 16.1 | **64.3** |

**Table 1:** Comparison of statistics across different datasets containing multi-turn trajectories of agent’s interactions with executable SWE environments.

*Note: Entries marked with asterisk (\*) indicate statistics whose values couldn't be derived from available data.*

For more details see our report in [Nebius blog](https://nebius.com/blog/posts/openhands-trajectories-with-qwen3-coder-480b).

---

# How to use
```python
import json

from datasets import load_dataset


ds = load_dataset("nebius/SWE-rebench-openhands-trajectories", split="train")

role2field_names = {
 "system": ["role", "content"],
 "assistant": ["role", "content", "tool_calls"],
 "user": ["role", "content"],
 "tool": ["role", "content", "name", "tool_call_id"],
}

def filter_and_deserialize(row):
    trajectory = []
    for msg in row["trajectory"]:
        msg = {
            field_name: msg[field_name] for field_name in role2field_names[msg["role"]]
        }
        if (msg["role"] == "assistant") and (msg["tool_calls"] is not None):
            for i, tool_call in enumerate(msg["tool_calls"]):
                if "arguments" in tool_call.get("function", {}):
                    msg["tool_calls"][i]["function"]["arguments"] = json.loads(
                        tool_call["function"]["arguments"]
                    )
        trajectory.append(msg)
    return row | {"trajectory": trajectory}

first_trajectory = filter_and_deserialize(ds[0])["trajectory"]

for msg in first_trajectory:
    print(msg)
```

---

# Dataset Structure
Each row contains the following information about trajectory:

| Field Name             | Type   | Description                                                                                             |
|------------------------|--------|---------------------------------------------------------------------------------------------------------|
| `trajectory_id`        | `str`  | The identifier unique for each collected trajectory.                                                    |
| `instance_id`          | `str`  | GitHub issue identifier consisting of repository name and issue number. Can be joined with corresponding Docker images from [nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench). |
| `repo`                 | `str`  | The repository identifier.                                                                              |
| `trajectory`           | `list` | Complete conversation history with roles: `'system'` (initial prompt), `'assistant'` (model reasoning/actions), `'user'` and `'tool'` (environment observations). |
| `model_patch`          | `str`  | Final code modifications produced by the agent in unified diff format.                                  |
| `exit_status`          | `str`  | Contains `'submit'` in case the agent completes trajectory with a terminating action, or an error message of the OpenHands agent otherwise. |
| `resolved`             | `int`  | Binary indicator of task success: `1` if the agent solved the issue, `0` otherwise.                     |
| `gen_tests_correct`    | `int`  | Number of agent-generated tests that correctly validate the solution (fail before applying the golden patch, pass after). `null` if no tests were generated. This metric **validates agent's test writing ability**. |
| `pred_passes_gen_test` | `int`  | Number of agent-generated tests passed by the agent's own solution (`model_patch`). `null` if no tests were generated. This metric **evaluates predicted solution correctness against the agent's own tests**. |

**Table 2:** Dataset field descriptions and schema.

To our knowledge, no other publicly available agent trajectory dataset includes evaluation of agent-generated tests.

**Important Note:** `arguments` field inside tool calls present in assistant steps of `trajectory` is serialized to string format for storage efficiency.
When training on this data, you may want to deserialize it first to ensure chat templates apply the same formatting during training and inference.

---

# Citation

```bibtex
@article{trofimova2025openhandstrajs,
  title={OpenHands Trajectories with Qwen3-Coder-480B-A35B-Instruct},
  author={Trofimova, Maria and Shevtsov, Anton and Ibragim, Badertdinov and Pyaev, Konstantin and Karasik, Simon and Golubev, Alexander},
  year={2025},
  journal={Nebius blog},
  note={}
}
```