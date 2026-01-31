# build runner image:
`docker build -t vllm-openai-pipeline:0.11.0 .` to build default image without proxy

`docker build --build-arg PROXY_SERVER="http://172.80.0.1:7890" -t vllm-openai-pipeline:0.11.0 .` to build image with default vllm version and custom proxy server

`docker build --build-arg VLLM_VERSION="v0.9.2" --build-arg PROXY_SERVER="http://172.80.0.1:7890" -t vllm-openai-pipeline:0.9.2-proxy .` to build image with custom vllm version and custom proxy server

# use runner:
## start container:
```
docker run \
        -itd \
        --gpus all \
        --network=host \
        --ipc=host \
        --cap-add IPC_LOCK \
        -e http_proxy="http://127.0.0.1:7890" \
        -e https_proxy="http://127.0.0.1:7890" \
        -v /home/models:/home/models \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /home/yanzhao/pipeline_results:/workspace/test_results \
        --name pipeline-host-docker-a \
        vllm-openai-pipeline:0.9.2-proxy
```

## attach container
`docker exec -it pipeline-host-docker-a /bin/bash`

## configure github runner
get connect token from github, run 

`./config.sh --url https://github.com/ModelEngine-Group/unified-cache-management --token ***`

## start runner
`sudo -E ./runner-svc.sh install`

`-E` here keeps the proxy setting env var.

## check runner log
`sudo ./runner-svc.sh status`

or go to `./_diag` to check more log