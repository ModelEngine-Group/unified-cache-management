#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
# Modified from 
# https://github.com/vllm-project/vllm-ascend/blob/main/csrc/build.sh

set -e

ROOT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

echo "ROOT_DIR: $ROOT_DIR"

cd $ROOT_DIR

IS_INSTALL="false"
for arg in "$@"; do
    if [[ "$arg" == "install" ]]; then
        IS_INSTALL="true"
    fi
done

mkdir -p build dist

rm -rf build/*
rm -rf dist/*

python3 setup_wheel.py build bdist_wheel

if [[ "$IS_INSTALL" == "true" ]]; then
    echo "Installing gsa_ops wheel ..." 
    pip3 install dist/gsa_ops*.whl --force-reinstall
fi

echo "Building finished. gsa_ops wheel is in $ROOT_DIR/dist"