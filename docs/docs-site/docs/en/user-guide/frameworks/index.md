# Deployment

Engine quickstarts connect one service to UCM. This section covers how to organize processes and cluster resources and where clients enter the service.

| Task | Guide | Main prerequisites |
| --- | --- | --- |
| Deploy a model with Docker | [Docker: Model Tour](../model-tour/index.md) | Model-specific images, devices, weights, and startup configuration |
| Manage model instances on Kubernetes | [Helm deployment](kubernetes/deploy.md) | Cluster dependencies, model configuration, images and storage |
| Use the pyMotor serving framework | [pyMotor](pyMotor.md) | Compatible Ascend runtime and framework settings |
| Run Prefill and Decode separately | [PD Disaggregation](../capabilities/pd-disaggregation/index.md) | Request coordination, a KV handoff path and compatible P/D instances |

Establish the target model and cache path with an [engine quickstart](../quick_start/index.md) before expanding. Use [operations](../observability/index.md) to check the request entry point, external reuse and backend state.
