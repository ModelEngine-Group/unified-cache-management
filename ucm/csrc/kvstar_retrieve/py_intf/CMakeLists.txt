find_library(
        TORCH_PYTHON_LIB_PATH
        torch_python
        HINTS ${TORCH_INSTALL_PREFIX}/lib
)

pybind11_add_module(
        kvstar_retrieve
        py_intf.cpp
)

target_link_libraries(
        kvstar_retrieve
        PRIVATE
        kvstar_retrieve.core
        ${TORCH_PYTHON_LIB_PATH}
        ${Torch_LIBRARIES}
)