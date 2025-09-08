# 找到 torch_python 库
find_library(
        TORCH_PYTHON_LIB_PATH
        torch_python
        HINTS ${TORCH_INSTALL_PREFIX}/lib
)

# 创建 Python 模块
pybind11_add_module(
        kvstar_retrieve
        py_intf.cpp
)

# 链接core静态库和torch_python。
# 因为core使用了PUBLIC，链接它的时候会自动继承它的头文件设置(即注入的EXTERNAL_INCLUDE_DIRS)。
target_link_libraries(
        kvstar_retrieve
        PRIVATE
        kvstar_retrieve.core
        ${TORCH_PYTHON_LIB_PATH}
        ${Torch_LIBRARIES}
)