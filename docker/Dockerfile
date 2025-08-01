# Set to other image if needed
FROM vllm/vllm-openai:v0.9.2

WORKDIR /workspace

# ReInstall vLLM for editting
RUN pip uninstall -y vllm && rm -rf /vllm-workspace/*
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_TAG=v0.9.2
RUN git clone --depth 1 $VLLM_REPO --branch $VLLM_TAG /vllm-workspace/vllm

# Set other VLLM_TARGET_DEVICE or other extra-index if needed
ENV VLLM_USE_PRECOMPILED=1
RUN VLLM_TARGET_DEVICE=cuda pip install -v -e /vllm-workspace/vllm --extra-index=https://download.pytorch.org/whl/nightly/cu128

# Install unified-cache-management
COPY . /vllm-workspace/unified-cache-management

RUN export PLATFORM="cuda" && \
     pip install -v -e /vllm-workspace/unified-cache-management

# Apply patch for vLLM
RUN cd /vllm-workspace/vllm \
    && git apply /vllm-workspace/unified-cache-management/unifiedcache/patch/vllm-adapt.patch

ENTRYPOINT ["/bin/bash"]