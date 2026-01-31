ARG VLLM_VERSION="v0.11.0"

FROM vllm/vllm-openai:${VLLM_VERSION}

ARG PROXY_SERVER=""
ENV http_proxy=$PROXY_SERVER
ENV https_proxy=$PROXY_SERVER

USER root

RUN useradd -m -s /bin/bash pipeline-runner && \
    echo "pipeline-runner ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/pipeline-runner && \
    chmod 0440 /etc/sudoers.d/pipeline-runner

WORKDIR /home/pipeline-runner

# set proxy for artifacts download, only affective for root user
RUN if [ -n "$PROXY_SERVER" ]; then \
      git config --global http.sslVerify false; \
    fi

RUN curl -o actions-runner-linux-x64-2.331.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.331.0/actions-runner-linux-x64-2.331.0.tar.gz

RUN tar xzf actions-runner-linux-x64-2.331.0.tar.gz && \
    chown -R pipeline-runner:pipeline-runner /home/pipeline-runner && \
    rm actions-runner-linux-x64-2.331.0.tar.gz

# rm after install to reduce image size
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages supervisor

RUN echo_supervisord_conf > /etc/supervisord.conf && \
    mkdir -p /var/log/supervisor && \
    mkdir -p /etc/supervisor/conf.d/ && \
    echo "" >> /etc/supervisord.conf && \
    echo "[include]" >> /etc/supervisord.conf && \
    echo "files = /etc/supervisor/conf.d/*.conf" >> /etc/supervisord.conf

COPY ./runner-svc.sh /home/pipeline-runner/runner-svc.sh

RUN sudo chmod a+rwx /home/pipeline-runner/runner-svc.sh

USER pipeline-runner

# set proxy for git in action runner, affective for pipeline-runner user
RUN if [ -n "$PROXY_SERVER" ]; then \
      git config --global http.sslVerify false; \
    fi

ENTRYPOINT ["/usr/local/bin/supervisord", "-n", "-c", "/etc/supervisord.conf"]