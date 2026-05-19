#!/bin/bash
set -x

BUILD_TYPE="DEBUG"
ARCH=$(uname -m)
PACKAGE_PLATFORM=${CUSTOM_PLAT_NAME:-"UnifiedCache-CUDA"}

while getopts "b:" arg; do
    case $arg in
        b)
            BUILD_TYPE=$OPTARG
            ;;
        ?)
            echo "Usage: $0 [-b <build_type>]"
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
export PLATFORM="cuda"

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
echo "Build version=$VERSION"

function check_build_install()
{
    cd ${KVCACHE_PROJECT_ROOT}
    pip3 install -r requirements.txt
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
        echo "Build wheels success"
        cd dist
        cp *.whl "$PACKAGE_DIR"
    else
        echo "Build wheels failed"
        exit 1
    fi
}

function delete_redundant_files() {
    rm -f *.whl
    rm -f version.ini
    rm -rf deployments/
    rm -rf docker/
    rm -f ucm_config.yaml
    rm -f metrics_configs.yaml
    rm -f *.patch
}

function collect_artifacts()
{
    cd ${PACKAGE_DIR}
    cp ${VERSION_FILE} .
    cp -r "${KVCACHE_PROJECT_ROOT}/docker" .
    cp -r "${KVCACHE_PROJECT_ROOT}/examples/deployments" .
    cp -r "${KVCACHE_PROJECT_ROOT}/examples/ucm_config_example.yaml" ucm_config.yaml
    cp -r "${KVCACHE_PROJECT_ROOT}/examples/metrics/metrics_configs.yaml" metrics_configs.yaml
    cp -r "${KVCACHE_PROJECT_ROOT}/test" .
    cp "${KVCACHE_PROJECT_ROOT}/ucm/integration/vllm/patch/0.11.0/vllm-adapt-sparse.patch" .
    cp "${KVCACHE_PROJECT_ROOT}/ucm/integration/sglang/patch/0.5.5/sglang-adapt.patch" .
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
