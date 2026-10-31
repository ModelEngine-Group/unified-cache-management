current_dir=$PWD
echo "当前工作目录是：$current_dir"

export PIP_INDEX_RUL="http://mirrors.tuna.tsinghua.edu.cn/pypi/web/sample"
export PLATFORM="ascend" ENABLE_SPARSE=true && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/ascend/Ascend/ascend-toolkit/latest/'uname -i'-linux/dev/LD_LIBRARY_PATH
pip install -v -e . --no-build-isolation

cd /vllm-workspace/vllm-ascend
git apply $current_dir/ucm/integration/vllm/patch/gsa/v0.23.0/vllm-ascend-csrc-build-1020-c8.patch
export SETUPTOOLS_SCM_PRETEND_VERSION="0.23.0"
pip install -v -e . --no-build-isolation

cd $current_dir