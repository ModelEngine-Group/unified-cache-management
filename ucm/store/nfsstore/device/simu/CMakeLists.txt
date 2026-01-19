add_library(storedevice STATIC simu_device.cc)
target_link_libraries(storedevice PUBLIC infra_status infra_logger)
