# Set to other image if needed
FROM vllm/vllm-openai:v0.9.2

WORKDIR /workspace

# Install unified-cache-management
COPY . /vllm-workspace/unified-cache-management

RUN export PLATFORM="cuda" && \
     pip install -v -e /vllm-workspace/unified-cache-management

# Apply patch for vLLM
RUN cd $(pip show vllm | grep Location | awk '{print $2}') \
    && git apply /vllm-workspace/unified-cache-management/unifiedcache/patch/0.9.2/vllm-adapt.patch

ENTRYPOINT ["/bin/bash"]