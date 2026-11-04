current_dir=$PWD
echo "当前工作目录是：$current_dir"

export PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
export PLATFORM="ascend" ENABLE_SPARSE=true && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/'uname -i'-linux/devlib
pip install -v -e . --no-build-isolation

cd /vllm-workspace/vllm
git checkout -- .

cd /vllm-workspace/vllm-ascend
git checkout -- .
git apply $current_dir/ucm/integration/vllm/patch/gsa/v0.23.0/vllm-ascend-csrc-build-1020-c8.patch
export SETUPTOOLS_SCM_PRETEND_VERSION="0.23.0"
pip install -v -e . --no-build-isolation

cd $current_dir