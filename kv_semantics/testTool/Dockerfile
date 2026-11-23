# testTool 开发环境镜像：Ubuntu + C++ 编译工具链
# 说明：ubuntu:24.04 为多架构镜像，x86_64 / arm64（如鲲鹏、其他 ARM 主机）均可直接构建
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        gdb \
        pkg-config \
        ca-certificates \
        curl \
        vim \
        less \
    && rm -rf /var/lib/apt/lists/*

# 宿主机 kv_semantics 目录运行时以 -v 挂载到 /workspace
RUN mkdir -p /workspace/testTool
WORKDIR /workspace/testTool

# 后台常驻，便于随时 docker exec 进入开发
CMD ["sleep", "infinity"]
