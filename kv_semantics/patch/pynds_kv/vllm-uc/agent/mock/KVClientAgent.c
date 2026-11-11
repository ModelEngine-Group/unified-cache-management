// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "KVClientAgent.h"

class agent::KVClientAgent::Private{};

namespace agent
{
    KVClientAgent::KVClientAgent() : private_impl(std::make_unique<Private>()) {}
    KVClientAgent::~KVClientAgent() = default;
    KVClientAgent::KVClientAgent(KVClientAgent&&) noexcept = default;
    KVClientAgent& KVClientAgent::operator = (KVClientAgent&&)noexcept = default;

    int KVClientAgent::init(const KVClientAgentConfig& config, const char* log_dir)
    {
        (void)config;
        (void)log_dir;
        return 0;
    }

    int KVClientAgent::start() { return 0; }

    uint64_t KVClientAgent::get_epoch() const { return 0; }

    std::vector<KVClientNodeView> KVClientAgent::get_view() const
    {
        return {
            {0, true,  true, {"192.168.133.197"}},
            {0, true,  true, {"192.168.133.198"}},
            {0, false, true, {"192.168.133.199"}},
        };
    }

    std::vector<KVClientNodeView> KVClientAgent::get_up_in_by_ns(const std::string& ns) const
    {
        (void)ns;
        return {
            {0, true,  true, {"192.168.133.197"}},
            {0, true,  true, {"192.168.133.198"}},
            {0, false, true, {"192.168.133.199"}},
        };
    }

    std::vector<KVClientNodeView> KVClientAgent::get_in() const
    {
        return {
            {0, true,  true, {"192.168.133.197"}},
            {0, true,  true, {"192.168.133.198"}},
            {0, false, true, {"192.168.133.199"}},
        };
    }

    std::vector<KVClientNodeView> KVClientAgent::get_in_by_ns(const std::string& ns) const
    {
        if (ns == "ns_01") {
            return {
                {0, true,  true, {"192.168.133.197"}},
                {0, true,  true, {"192.168.133.198"}},
                {0, false, true, {"192.168.133.199"}},
            };
        } else if (ns == "ns_02") {
            return {
                {0, true,  true, {"192.168.133.198"}},
                {1, true,  true, {"192.168.133.198"}},
                {2, false, true, {"192.168.133.198"}},
            };
        } else {
            return {
                {0, true,  true, {"192.168.133.199"}},
                {1, true,  true, {"192.168.133.199"}},
                {2, false, true, {"192.168.133.199"}},
            };
        }
    }

    std::vector<std::string> KVClientAgent::get_all_ns_name() const { return {"ns_01"}; }

}  // namespace agent
