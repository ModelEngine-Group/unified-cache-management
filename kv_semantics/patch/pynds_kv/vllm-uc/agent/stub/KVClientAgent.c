// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "KVClientAgent.h"
#include <cstdio>

// Non-cluster builds do not call the agent. Keep the same ABI surface so the
// SDK links without vendor libraries; fail clearly if cluster mode is selected.
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
        std::fprintf(stderr,
                     "[NDS] cluster mode requires the vendor agent libraries; "
                     "use sceneMode=0 or rebuild with AGENT_MODE=release.\n");
        return -1;
    }

    int KVClientAgent::start() { return -1; }

    int KVClientAgent::start_view_refresh() { return -1; }

    int KVClientAgent::stop_view_refresh() { return -1; }

    uint64_t KVClientAgent::get_epoch() const { return 0; }

    std::vector<KVClientNodeView> KVClientAgent::get_view() const { return {}; }

    std::vector<KVClientNodeView> KVClientAgent::get_view_by_ns(const std::string& ns, bool up,
                                                                bool in) const
    {
        (void)ns;
        (void)up;
        (void)in;
        return {};
    }

    std::vector<KVClientNodeView> KVClientAgent::get_up_in_by_ns(const std::string& ns) const
    {
        (void)ns;
        return {};
    }

    std::vector<KVClientNodeView> KVClientAgent::get_down_in_by_ns(const std::string& ns) const
    {
        (void)ns;
        return {};
    }

    std::vector<KVClientNodeView> KVClientAgent::get_up_out_by_ns(const std::string& ns) const
    {
        (void)ns;
        return {};
    }

    std::vector<KVClientNodeView> KVClientAgent::get_down_out_by_ns(const std::string& ns) const
    {
        (void)ns;
        return {};
    }

    std::vector<KVClientNodeView> KVClientAgent::get_in() const { return {}; }

    std::vector<KVClientNodeView> KVClientAgent::get_in_by_ns(const std::string& ns) const
    {
        (void)ns;
        return {};
    }

    std::vector<std::string> KVClientAgent::get_all_ns_name() const { return {}; }

}  // namespace agent
