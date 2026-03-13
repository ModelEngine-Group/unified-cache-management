#!/usr/bin/env bash
# Build and install ucm_custom_ops so that in Python you can:
#   import ucm_custom_ops
#   torch.ops._C_ucm.npu_reshape_and_cache_bnsd(...)
#   torch.ops._C_ucm.npu_hamming_dist_top_k(...)

ROOT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

echo "ROOT_DIR: $ROOT_DIR"

# install ucm_custom_ops python package
cd $ROOT_DIR

# ensure build/dist exist before cleaning
mkdir -p build dist
# clean build and dist directories
rm -rf build/*
rm -rf dist/*

# build ucm_custom_ops python package
python3 setup_wheel.py build bdist_wheel
