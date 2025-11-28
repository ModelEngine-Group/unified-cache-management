export DEVICE=/dev/davinci7
export MODEL_PATH=/home/models/Qwen3-0.6B

docker run -d \
    --name ucm_ascend_demo \
    --network=host \
    --device=$DEVICE \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v $MODEL_PATH:$MODEL_PATH \
    ucm_ascend:1110 \
    vllm serve $MODEL_PATH --trust-remote-code