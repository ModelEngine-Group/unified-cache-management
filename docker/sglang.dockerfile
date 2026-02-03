# Set to other image if needed
FROM lmsysorg/sglang:v0.5.5.post3

ARG PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

WORKDIR /workspace

# Install unified-cache-management
COPY . /workspace/unified-cache-management

RUN pip config set global.index-url ${PIP_INDEX_URL}

RUN export PLATFORM="cuda" ENABLE_SPARSE=false && \
     pip install -v -e /workspace/unified-cache-management --no-build-isolation

# Apply patch for SGLang
RUN cd $(pip show sglang | grep Location | awk '{print $2}') \
    && git apply /workspace/unified-cache-management/ucm/integration/sglang/patch/0.5.5/sglang-adapt.patch

ENTRYPOINT ["/bin/bash"]
