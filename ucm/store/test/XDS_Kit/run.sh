#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

SHORT=v:,
LONG=soc-version:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
SOC_VERSION="Ascend910B3"

SOURCE_FILES="XDS_Kit.cc"
LOCAL_LIB="./libndsfs.so"
OUTPUT_EXECUTABLE="XDS_Kit"

while :; do
    case "$1" in
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
echo "Current compile soc version is ${SOC_VERSION}"

rm -rf build
mkdir -p build
cmake -B build \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
cmake --build build -j
cmake --install build
