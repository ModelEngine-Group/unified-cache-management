KVStar Sparse Attention Retrieve Algorithm need install async retrieve c++ lib

cd async_retrieve_lib

pip install -v -e .

then config in kv_connector_extra_config

kv_connector_extra_config={
            "ucm_sparse_method": "KVStarMultiStep",
        }

