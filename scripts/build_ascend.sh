#!/bin/bash
set -x

BUILD_TYPE="DEBUG"
ASCEND_PLATFORM="ascend"
ASCEND_HOME="/usr/local/Ascend"
PACKAGE_PLATFORM=${CUSTOM_PLAT_NAME:-$ASCEND_PLATFORM}
ARCH=$(uname -m)

while getopts "b:p:" arg; do
    case $arg in
        b)
        
            BUILD_TYPE=$OPTARG
            ;;
        p)
            ASCEND_PLATFORM=$OPTARG
            ;;
        ?)
            echo "Usage: $0 [-b <build_type>] [-p <ascend/ascend-a3>]"
            exit 0
            ;;
    esac
done

PROJECT_ROOT="$WORKSPACE"
KVCACHE_PROJECT_ROOT="$PROJECT_ROOT/unified-cache-management"
PACKAGE_DIR="$PROJECT_ROOT/package"
if [ ! -d ${PACKAGE_DIR} ]; then
    mkdir ${PACKAGE_DIR}
fi
export PLATFORM="$ASCEND_PLATFORM"

VERSION_FILE="${PROJECT_ROOT}/version.ini"
if [ ! -f "${VERSION_FILE}" ]; then
    VERSION_FILE="${KVCACHE_PROJECT_ROOT}/version.ini"
fi
if [ ! -f "${VERSION_FILE}" ]; then
    echo "version.ini does not exist."
    exit 1
fi
. ${VERSION_FILE}
VERSION=$VLLM_UC_VERSION

source "$ASCEND_HOME/ascend-toolkit/set_env.sh" || echo "0"
source "$ASCEND_HOME/nnal/atb/set_env.sh" || echo "0"
if [ -z "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/aarch64-linux/devlib
else
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ASCEND_HOME/ascend-toolkit/latest/aarch64-linux/devlib
fi
echo "Build version=$VERSION, platform=$PLATFORM, ASCEND_HOME_PATH=$ASCEND_HOME_PATH, ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME, LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

function check_build_install()
{
    cd ${KVCACHE_PROJECT_ROOT}
    # pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "requirements installation failed, please check the network and environment configurations"
        exit 1
    else
        echo "requirements are successfully installed"
    fi
}

function build_wheels()
{
    cd ${PACKAGE_DIR}
    rm -f *.whl

    # switch off DOWNLOAD_DEPENDENCE
    # sed -i '/DOWNLOAD_DEPENDENCE/s/ON/OFF/' ${KVCACHE_PROJECT_ROOT}/CMakeLists.txt

    cd ${KVCACHE_PROJECT_ROOT}
    export ENABLE_SPARSE=${ENABLE_SPARSE:-TRUE}
    python3 -m build --no-isolation --wheel

    if [ $? -eq 0 ]; then
        echo "Build ucm success"
        cp ucm/sparse/gsa_on_device/csrc/ascend/dist/*.whl "$PACKAGE_DIR"
        cp ucm/sparse/gsa_on_device/csrc/ascend/output/*.run "$PACKAGE_DIR"
        cp dist/*.whl "$PACKAGE_DIR"
    else
        echo "Build ucm failed"
        exit 1
    fi
}

function delete_redundant_files() {
    rm -f *.whl
    rm -f version.ini
    rm -rf deployments/
    rm -rf docker/
    rm -rf deploy
    rm -f *.run
    rm -f *.patch
    rm -f install.sh
    rm -f install_ascend_ops.sh
}

function collect_artifacts()
{
    cd ${PACKAGE_DIR}
    cp ${VERSION_FILE} .
    mkdir -p deploy/script
    cp "${KVCACHE_PROJECT_ROOT}/install.sh" .
    cp -r "${KVCACHE_PROJECT_ROOT}/docker" .
    cp -r "${KVCACHE_PROJECT_ROOT}/examples/deployments" .
    cp -r "${KVCACHE_PROJECT_ROOT}/examples/ucm_config_example.yaml" deploy/script/ucm_config.yaml
    cp -r "${KVCACHE_PROJECT_ROOT}/examples/metrics/metrics_configs.yaml" deploy/script/metrics_configs.yaml
    cp -r "${KVCACHE_PROJECT_ROOT}/test" .
    cp "${KVCACHE_PROJECT_ROOT}/ucm/store/test/e2e/posixstore_aio_test.py" ./test/
    cp "${KVCACHE_PROJECT_ROOT}/ucm/integration/vllm/patch/0.11.0/vllm-adapt.patch" .
    cp "${KVCACHE_PROJECT_ROOT}/ucm/integration/vllm/patch/0.11.0/vllm-ascend-adapt.patch" .
    cp "${KVCACHE_PROJECT_ROOT}/ucm/sparse/gsa_on_device/csrc/ascend/install.sh" install_ascend_ops.sh
}

function package_all()
{
    collect_artifacts
    if [ "${SKIP_TAR}" != "1" ]; then
        cd ${PACKAGE_DIR}
        tar -czvf "AI-Storage-Kit_${VERSION}_${PACKAGE_PLATFORM^^}_${ARCH}_${BUILD_TYPE}.tar.gz" *
        delete_redundant_files
    fi
}

check_build_install
build_wheels
package_all
