#pragma once

#include <cstdint>
#include <map>
#include <string>
#include "core/transport.h"

namespace transport {

struct HixlInitAttrs : public InitAttrs {
    std::string local_engine;
    std::map<std::string, std::string> options;
    int device_id = -1;
    int32_t connect_timeout_ms = 1000;
    int32_t transfer_timeout_ms = 1000;
};

}  // namespace transport
