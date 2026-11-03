/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: KV Client Agent 实现
 */

// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#include "KVClientAgent.h"
#include <cerrno>
#include <utility>
#include "AgentLogger.h"
#include "KVClientAgentImpl.h"


namespace agent {

class KVClientAgent::Private {
public:
    Private() : impl(create_kv_client_agent()) {}

    AgentLogger logger;
    std::unique_ptr<KVClientAgentImpl> impl;
};

KVClientAgent::KVClientAgent() : private_impl(std::make_unique<Private>()) {}

KVClientAgent::~KVClientAgent() = default;
KVClientAgent::KVClientAgent(KVClientAgent&&) noexcept = default;
KVClientAgent& KVClientAgent::operator=(KVClientAgent&&) noexcept = default;

int KVClientAgent::init(const KVClientAgentConfig& config, const char* log_dir)
{
    if (!private_impl || !private_impl->impl) { return -ENOMEM; }
    if (config.mon_host.empty() || config.client_name.empty() || config.client_key.empty()) {
        return -EINVAL;
    }

    int r = private_impl->logger.init(log_dir, "kv_client_agent");
    if (r < 0) { return r; }
    return private_impl->impl->init(config, &private_impl->logger);
}

int KVClientAgent::start()
{
    if (!private_impl || !private_impl->impl) { return -ENOMEM; }
    return private_impl->impl->start();
}

uint64_t KVClientAgent::get_epoch() const
{
    if (!private_impl || !private_impl->impl) { return 0; }
    return private_impl->impl->get_epoch();
}

std::vector<KVClientNodeView> KVClientAgent::get_view() const
{
    if (!private_impl || !private_impl->impl) { return {}; }
    return private_impl->impl->get_view();
}

std::vector<KVClientNodeView> KVClientAgent::get_view_by_ns(const std::string& ns, bool up,
                                                            bool in) const
{
    if (!private_impl || !private_impl->impl) { return {}; }
    return private_impl->impl->get_view_by_ns(ns, up, in);
}

std::vector<KVClientNodeView> KVClientAgent::get_up_in_by_ns(const std::string& ns) const
{
    if (!private_impl || !private_impl->impl) { return {}; }
    return private_impl->impl->get_up_in_by_ns(ns);
}

std::vector<KVClientNodeView> KVClientAgent::get_down_in_by_ns(const std::string& ns) const
{
    if (!private_impl || !private_impl->impl) { return {}; }
    return private_impl->impl->get_down_in_by_ns(ns);
}

std::vector<KVClientNodeView> KVClientAgent::get_up_out_by_ns(const std::string& ns) const
{
    if (!private_impl || !private_impl->impl) { return {}; }
    return private_impl->impl->get_up_out_by_ns(ns);
}

std::vector<KVClientNodeView> KVClientAgent::get_down_out_by_ns(const std::string& ns) const
{
    if (!private_impl || !private_impl->impl) { return {}; }
    return private_impl->impl->get_down_out_by_ns(ns);
}

std::vector<KVClientNodeView> KVClientAgent::get_in() const
{
    if (!private_impl || !private_impl->impl) { return {}; }
    return private_impl->impl->get_in();
}

std::vector<KVClientNodeView> KVClientAgent::get_in_by_ns(const std::string& ns) const
{
    if (!private_impl || !private_impl->impl) { return {}; }
    return private_impl->impl->get_in_by_ns(ns);
}

std::vector<std::string> KVClientAgent::get_all_ns_name() const
{
    if (!private_impl || !private_impl->impl) { return {}; }
    return private_impl->impl->get_all_ns_names();
}

int KVClientAgent::start_view_refresh()
{
    if (!private_impl || !private_impl->impl) { return -ENOMEM; }
    return private_impl->impl->start_view_refresh();
}

int KVClientAgent::stop_view_refresh()
{
    if (!private_impl || !private_impl->impl) { return -ENOMEM; }
    return private_impl->impl->stop_view_refresh();
}

}  // namespace agent
