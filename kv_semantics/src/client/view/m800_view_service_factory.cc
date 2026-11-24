/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include "view/m800_view_service_factory.h"
#include <string>
#include <utility>
#include "parser_common.h"
#include "view/m800_view_common.h"
#include "view/m800_view_master_server.h"
#include "view/m800_view_slave_server.h"

namespace kv {
namespace {

class M800ViewConfigErrorViewServer final : public ViewServer {
public:
    explicit M800ViewConfigErrorViewServer(Status status) : status_(std::move(status)) {}

    Status GetGlobalView(GlobalView& view) override
    {
        (void)view;
        return status_;
    }

private:
    Status status_;
};

Status FactoryError(StatusCode code, const std::string& message)
{
    return Status::Error(code, message);
}

}  // namespace

bool HasViewProvider(const KvClientConfig& config)
{
    return config.attrs.find("view.provider") != config.attrs.end();
}

bool IsM800ViewProvider(const KvClientConfig& config)
{
    const auto iter = config.attrs.find("view.provider");
    return iter != config.attrs.end() && ToLower(TrimConfigValue(iter->second)) == "m800";
}

std::shared_ptr<ViewServer> CreateM800ViewServer(const KvClientConfig& config)
{
    if (!IsM800ViewProvider(config)) {
        const auto iter = config.attrs.find("view.provider");
        const auto value =
            iter == config.attrs.end() ? std::string{} : TrimConfigValue(iter->second);
        return std::make_shared<M800ViewConfigErrorViewServer>(FactoryError(
            StatusCode::INVALID_ARGUMENT, "view.provider must be m800, value=" + value));
    }

    M800ViewServerConfig m800Config;
    auto status = M800ViewServerConfig::Build(config, m800Config);
    if (!status.ok()) { return std::make_shared<M800ViewConfigErrorViewServer>(status); }

    if (m800Config.role == M800ViewRole::MASTER) {
        auto master = CreateM800ViewMasterServer(m800Config);
        if (master == nullptr) {
            return std::make_shared<M800ViewConfigErrorViewServer>(
                FactoryError(StatusCode::INTERNAL_ERROR, "failed to allocate M800 view master"));
        }
        return master;
    }

    auto slave = CreateM800ViewSlaveServer(m800Config);
    if (slave == nullptr) {
        return std::make_shared<M800ViewConfigErrorViewServer>(
            FactoryError(StatusCode::INTERNAL_ERROR, "failed to allocate M800 view slave"));
    }
    return slave;
}

}  // namespace kv
