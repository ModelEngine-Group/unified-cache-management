add_library(device OBJECT simu_device.cc)
target_link_libraries(device PUBLIC infra)
