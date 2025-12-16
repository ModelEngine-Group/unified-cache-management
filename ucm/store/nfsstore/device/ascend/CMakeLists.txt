set(ASCEND_ROOT "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Path to Ascend root directory")
add_library(Ascend::ascendcl UNKNOWN IMPORTED)
set_target_properties(Ascend::ascendcl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_ROOT}/include"
    IMPORTED_LOCATION "${ASCEND_ROOT}/lib64/libascendcl.so"
)

add_library(storedevice STATIC ascend_device.cc)
target_link_libraries(storedevice PUBLIC infra_status Ascend::ascendcl)
