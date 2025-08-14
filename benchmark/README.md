# Performance Benchmark

***

This document shows how to measure the performance (and accuracy, coming soon) of unified-cache-manager integrated with LLM API server.

Our dataset includes doc-qa questions with context lengths of 2k, 4k, 8k, 16k, 32k and 64k, each corresponding to a jsonl file under the directory src/llmperf.

**Benchmark Coverage**: We measure e2e latency and throughput, together with TTFT, TPOT, etc. TTFT is the key metric that we emphasize on now. 

### 1. Create a new conda environment to run our performance benchmark

***

#### Basic requirements

***

+ python >= 3.8 and <= 3.10, 3.10 is recommended

Run
```
pip install -e .
```

to install the llmperf library by source code on your environment.

### 2. Run the test script

***

```
bash doc-qa.sh -m <path-to-your-model> -a <served-model-name> -c <connector> -d <result-output-directory> -l <context-lengths> -i <llm-server-ip> -p <llm-server-port> -b <compute-device-backend> [-s <special delimiter string in CacheBlend>]
```

#### Arguments explanations

+ **-m**: Path to your model.
+ **-a**: Served model name while launching the LLM server. If not specified while launching the server, it should be equal to path-to-your-model (-m).
+ **-c**: Connector that llm server uses. Can be any string, like "NFS" or "DRAM".
+ **-d**: Path to save the performance results.
+ **-l**: All context lengths to measure the performance (**Unit: k tokens**). Separate different lengths with commas. For example, you can set "**-l 2,4,8**" if you want to measure the performance on context lengths of 2k, 4k and 8k.
+ **-i**, **-p**: IP address and port of your LLM server.
+ **-b**: Compute device backend. Can be any string, like "NPU" or "GPU".
+ **-s**: (Optional) Cacheblend split string. Should align with the LMCACHE_BLEND_SPECIAL_STR in the settings of CacheBlend.

#### Examples

```
bash doc-qa.sh /home/models/QwQ-32B -a /home/models/QwQ-32B -c DRAM -d result_outputs -l 2,4,8 -i localhost -p 8000 -b NPU
```

The above command evaluates the performance of model QwQ-32B with model source files under /home/models/QwQ-32B, under context lengths of 2k, 4k and 8k, using the DRAM-connector of UCManager with computing device as Ascend NPU. The LLM server is deployed on localhost listening to port 8000, and the served-model-name is not specified while launching the server. The performance results will be saved under directory result_outputs after the script finishes running.

```
bash doc-qa.sh /home/models/QwQ-32B -a QWQ -c NFS -d result_outputs -l 2,4,8,16,32 -i 141.111.32.68 -p 9999 -b GPU-L40 -s " # # "
```

The above command evaluates the performance of model QwQ-32B with model source files under /home/models/QwQ-32B, under context lengths of 2k, 4k, 8k, 16k and 32k, using the NFS-connector of UCManager with computing device as Nvidia-GPU-L40. The LLM server is deployed on 141.111.32.68 listening to port 9999, and the served-model-name is specified as "QWQ" while launching the server. The performance results will be saved under directory result_outputs after the script finishes running. It uses the feature of CacheBlend with special dilimeter string " # # ", which should be aligned with LMCACHE_BLEND_SPECIAL_STR in the settings of CacheBlend.


### 3. Performance results

***

The context_length-TTFT figure will be saved automatically under the {result-output-directory} you just set. The filename is organized in the format

```
docqa_TTFT_{path-to-your-model}_{connector}_{compute-device-backend}
```

Below shows the results we obtained using NFS connector on Nvidia-L40-GPU (left) and Ascend-NPU (right) backends with context lengths of 2k, 4k, 8k, 16k, 32k.


![](images/GPU_NPU_0813.png)