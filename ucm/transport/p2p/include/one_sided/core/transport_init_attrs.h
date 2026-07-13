#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "core/transport.h"

namespace transport {

struct HixlInitAttrs : public InitAttrs {
    struct Instance {
        int32_t port = 0;
        int32_t device_id = -1;
        std::map<std::string, std::string> options;
    };

    std::string ip = "127.0.0.1";
    std::vector<Instance> instances;
    int32_t connect_timeout_ms = 1000;
    int32_t transfer_timeout_ms = 1000;
};

}  // namespace transport
