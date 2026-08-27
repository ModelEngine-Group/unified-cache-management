# pyMotor

**pyMotor** (MindIE-Motor) is an Ascend-native distributed inference serving
framework that runs vLLM-style prefill/decode (PD) disaggregation on Atlas
hardware. A pyMotor cluster is described as Kubernetes-style configs — a
Coordinator plus Engine Pods driven by `user_config.json` and `deploy.py` —
which makes deployment, scaling, and failover declared and reproducible.

pyMotor serves vLLM-compatible engines on Ascend and ships with features such
as manual scaling, primary/standby failover, request tracing, PD-role
rescheduling, and container snapshot.

UCM plugs into pyMotor as a persistent KV cache store backend
(`UCMConnector` / `UcmPipelineStore`): prefill writes the KVCache once, and
later requests that share the same prefix reuse it instead of recomputing.

## Deployment

Installation and deployment are covered by the official MindIE-Motor
documentation:

| Task | Guide |
| --- | --- |
| Documentation home | [MindIE-Motor docs](https://mindie-motor.readthedocs.io/zh-cn/latest/) |
| Environment preparation | [Official guide](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/environment_preparation/) |
| Quick start | [Official guide](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/quick_start_motor/) |
| Kubernetes deployment — PD disaggregation | [Official guide](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/deployment/k8s/pd_disaggregation_deployment/) |
| Kubernetes deployment — PD aggregation | [Official guide](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/deployment/k8s/pd_aggregation_deployment/) |
| Standalone Coordinator deployment | [Official guide](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/deployment/standalone/) |

## UCM as the KV cache store backend

To run pyMotor with UCM as its persistent KV cache store, follow the dedicated
integration guide:

- [UCM backend for pyMotor KV cache store](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/features/kv_cache_store/backend/ucm/)

It walks through installing the UCM wheel in the prefill role, mounting the
shared cache storage under `motor_deploy_config.storage`, wiring the
`MultiConnector` (Mooncake in front, `UCMConnector` in position 2) for prefill,
and verifying KVCache hits on repeat requests.