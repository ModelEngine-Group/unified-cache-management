#pragma once

#include <memory>
#include <string>
#include "kv_test_types.h"

namespace UC::KVTest {

class AsuRuntimeProxy {
public:
    static AsuRuntimeProxy& Instance();

    Status Load(const AsuRuntimeLibraryConfig& config);
    UC::ASU::Status LoadAsuClientConfig(const std::string& configPath,
                                        UC::ASU::AsuClientConfig& config);
    std::unique_ptr<UC::ASU::AsuClient> CreateAsuClient(
        const UC::ASU::TransportFactory* transportFactory, Status& status);
    std::unique_ptr<UC::ASU::AsuTransport> CreateAsuTransport(Status& status);

private:
    AsuRuntimeProxy() = default;

    Status EnsureLoaded();

    using CreateClientFn =
        std::unique_ptr<UC::ASU::AsuClient> (*)(const UC::ASU::TransportFactory*);
    using CreateTransportFn = std::unique_ptr<UC::ASU::AsuTransport> (*)();
    using LoadClientConfigFn = UC::ASU::Status (*)(const char*, UC::ASU::AsuClientConfig*);

    void* clientHandle_{nullptr};
    void* transportHandle_{nullptr};
    CreateClientFn createClient_{nullptr};
    CreateTransportFn createTransport_{nullptr};
    LoadClientConfigFn loadClientConfig_{nullptr};
    AsuRuntimeLibraryConfig config_;
};

}  // namespace UC::KVTest
