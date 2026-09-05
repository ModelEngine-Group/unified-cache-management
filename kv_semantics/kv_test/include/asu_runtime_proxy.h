#pragma once

#include <memory>
#include <string>
#include "kv_client_impl.h"
#include "kv_test_types.h"
#include "kv_transport.h"

namespace kv::bench {

class AsuRuntimeProxy {
public:
    static AsuRuntimeProxy& Instance();

    Status Load(const AsuRuntimeLibraryConfig& config);
    kv::Status LoadAsuClientConfig(const std::string& configPath,
                                        kv::AsuClientConfig& config);
    std::unique_ptr<kv::AsuClient> CreateAsuClient(const kv::TransportFactory* transportFactory,
                                                    Status& status);
    std::unique_ptr<kv::AsuTransport> CreateAsuTransport(Status& status);

private:
    AsuRuntimeProxy() = default;

    Status EnsureLoaded();

    using CreateClientFn = std::unique_ptr<kv::AsuClient> (*)(const kv::TransportFactory*);
    using CreateTransportFn = std::unique_ptr<kv::AsuTransport> (*)();
    using LoadClientConfigFn = kv::Status (*)(const char*, kv::AsuClientConfig*);

    void* clientHandle_{nullptr};
    void* transportHandle_{nullptr};
    CreateClientFn createClient_{nullptr};
    CreateTransportFn createTransport_{nullptr};
    LoadClientConfigFn loadClientConfig_{nullptr};
    AsuRuntimeLibraryConfig config_;
};

}  // namespace kv::bench
