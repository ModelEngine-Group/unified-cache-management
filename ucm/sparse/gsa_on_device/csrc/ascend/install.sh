#!/usr/bin/env bash

ROOT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
WITH_SOURCE="false"

while getopts "s" arg; do
    case $arg in
        s)
            WITH_SOURCE="true"
            ;;
        ?)
            echo "Usage: $0 [-s]"
            echo "  -s: Install with source code"
            exit 1
            ;;
    esac
done

echo "ROOT_DIR: $ROOT_DIR, WITH_SOURCE: $WITH_SOURCE"

RUN_FILE_DIR=$ROOT_DIR
WHL_FILE_DIR=$ROOT_DIR
if [ "$WITH_SOURCE" = "true" ]; then
    RUN_FILE_DIR="$ROOT_DIR/output"
    WHL_FILE_DIR="$ROOT_DIR/dist"
fi

# install custom_ops in the default path /usr/local/Ascend/latest/opp/vendors
echo "Installing custom ops in /usr/local/Ascend/latest/opp/vendors"
cd $RUN_FILE_DIR
./UCM-custom_ops*.run

# update environment variables LD_LIBRARY_PATH
line="export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/ucm/op_api/lib/:\${LD_LIBRARY_PATH}"
env_file="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/ucm/setenv.bash"
if [ ! -f "$env_file" ]; then
    touch "$env_file"
fi

if [ -n "$line" ] && ! grep -Fqx "$line" ~/.bashrc 2>/dev/null; then
    echo "$line" >> "$env_file"
    echo "$line" >> ~/.bashrc
fi
source "$env_file"

# install ucm_custom_ops python package
cd $WHL_FILE_DIR
pip3 install ucm_custom_ops*.whl --force-reinstall

