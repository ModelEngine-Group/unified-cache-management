
ROOT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

echo "ROOT_DIR: $ROOT_DIR"

# install ucm_custom_ops python package
cd $ROOT_DIR

# clean build and dist directories
rm -rf build/*
rm -rf dist/*

# uninstall ucm_custom_ops python package
pip3 uninstall ucm_custom_ops -y

# build ucm_custom_ops python package
python3 setup_wheel.py build bdist_wheel

# install ucm_custom_ops python package
cd $ROOT_DIR/dist
pip3 install ucm_custom_ops*.whl --force-reinstall