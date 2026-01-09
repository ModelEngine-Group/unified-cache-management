# Single-Machine Deployment (CUDA or Ascend)

This scenario applies to a single physical server and uses two files:  
- `vllm/config.properties`
- `vllm/run_vllm.sh`

Modify the parameters in `config.properties` according to your actual requirements (e.g., model, memory).  

**Note:** `Multi-node Configuration`, `Ray Configuration` and `Ascend Multi-node Data Parallel` **can be ignored**, as they are only used in multi-machine inference scenarios.

After completing the configuration, launch the service with:
```bash
bash run_vllm.sh
```

# Multi-Machine Deployment (CUDA)
In multi-node CUDA deployments, vLLM relies on Ray as its distributed backend. Therefore, in addition to `vllm/config.properties` and `vllm/run_vllm.sh`, you must also use `vllm/start_ray.sh` to start the Ray cluster. For a two-node deployment, follow these steps:

step1 Modify config.properties
- Set `master_ip` to the IP address of the head node
- Set `worker_ip` to the IP address of the worker node
- Set `node_num` to 2
- Set `distributed_executor_backend` to `ray`
- `Ascend Multi-node Data Parallelism` **can be ignored**, as it is only used in Ascend multi-machine data parallelism inference scenarios.
- Adjust other vLLM parameters as needed

step2 Start the Ray cluster
- On the head node:
    ```bash
    NODE=0 bash start_ray.sh
    ```
- On the worker node:
    ```bash
    NODE=1 bash start_ray.sh
    ```

step3 Launch the vLLM service

Run the following command on **either node**:
```bash
bash run_vllm.sh
```

**Scaling Note:**  To deploy across more machines, set `node_num` to the actual number of nodes and ensure that each worker node’s `worker_ip` is configured to its own IP address.

# Multi-Machine Deployment (Ascend)

Ascend multi-node deployments differ based on whether **Data Parallelism (DP)** is enabled.

## Case 1: DP = 1 (No Data Parallelism)

This case follows the same procedure as CUDA multi-machine deployment and requires the following files:
- `vllm/config.properties`
- `vllm/run_vllm.sh`
- `start_ray.sh`

Follow the exact steps described in the **CUDA Multi-Machine Deployment** section above.

## Case 2: DP > 1 (Data Parallelism Enabled)
This scenario requires the following files:
- `vllm/config.properties`
- `vllm/run_vllm_dp.sh`
  
For a two-node deployment, follow these steps:

step1 Modify `config.properties`
- Set `master_ip` and `worker_ip`
- Set `dp_size_local` to the number of DP per node
- `Ray Configuration` can be ignored, as it is not used in the current scenario.
- Adjust other vLLM parameters as needed

step2 Launch the vLLM service
- On the head node:
    ```bash
    NODE=0 bash run_vllm_dp.sh
    ```
- On the worker node:
    ```bash
    NODE=1 bash run_vllm_dp.sh
    ```

**Scaling Note:** When deploying across more nodes, ensure that each worker node’s `worker_ip` is correctly set to its local IP address.
