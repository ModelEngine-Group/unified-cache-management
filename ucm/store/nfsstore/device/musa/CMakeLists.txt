set(MUSA_ROOT "/usr/local/musa/" CACHE PATH "Path to MUSA root directory")
add_library(Musa::musart UNKNOWN IMPORTED)
set_target_properties(Musa::musart PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MUSA_ROOT}/include"
    IMPORTED_LOCATION "${MUSA_ROOT}/lib/libmusart.so"
)

add_library(storedevice STATIC musa_device.cc)
target_link_libraries(storedevice PUBLIC infra_status Musa::musart)
