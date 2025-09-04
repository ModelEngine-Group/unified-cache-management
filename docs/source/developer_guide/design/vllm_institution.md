# vLLM Institution
This doc shows how the UcmKVStoreBase work with KV connector api in v1 to support multiple types of backends to expand storage for users. The details of KV connector api in v1 refer to this pr : [KV Connector API V1](https://github.com/vllm-project/vllm/pull/15960).
## How it works
As you can see in the README part, the KVStoreBase helps decoupling sparse algorithms and external storage, a class that inherits from KVConnectorBase_V1 named UnifiedCacheConnectorV1 facilitates the connection between vLLM v1 and this class, the The figure below shows how it worked:

![uc_connector](../../images/ucconn_ucmconn.png)(../../images/ucconn_ucmconn.png)

The interfaces designed in KVStoreBase are similar to the KV connector API in v1, which are divided into scheduler-side methods and worker-side methods, as follows:
- scheduler methods
  - `lookup`: Look up KV blocks stored in external storage by vLLM block hash.
  - `create`: Initialize kv cache space in external storage.
  - `commit`: Mark cache operations as complete or fail and ready for reuse.
- worker methods
  - `load`: Load KV cache blocks from external storage back to HBM.
  - `dump`: Offload KV cache blocks from HBM to external storage.
  - `wait`: Ensure that all dump or load operations have completed.
  
## Example
If you'd like to run the use case to gain a better understanding, please click ***Getting Started -> Example*** for more details.