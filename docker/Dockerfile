# Set to other image if needed
FROM vllm/vllm-openai:v0.9.2

ARG PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

WORKDIR /workspace

# Install unified-cache-management
COPY . /workspace/unified-cache-management

RUN pip config set global.index-url ${PIP_INDEX_URL}

RUN export PLATFORM="cuda" ENABLE_SPARSE=true && \
     pip install -v -e /workspace/unified-cache-management --no-build-isolation

# Apply patch for vLLM
RUN cd $(pip show vllm | grep Location | awk '{print $2}') \
    && git apply /workspace/unified-cache-management/ucm/integration/vllm/patch/0.9.2/vllm-adapt.patch

ENTRYPOINT ["/bin/bash"]