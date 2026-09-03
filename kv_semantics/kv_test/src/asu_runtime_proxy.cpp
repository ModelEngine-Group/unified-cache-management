#include "asu_runtime_proxy.h"
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <limits.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace UC::KVTest {

namespace {

constexpr int kExitInvalidArgument = 1;
constexpr const char* kClientLibraryEnv = "KV_TEST_ASU_CLIENT_LIB";
constexpr const char* kTransportLibraryEnv = "KV_TEST_ASU_TRANSPORT_LIB";
constexpr const char* kDefaultClientLibrary = "libasu_client.so";
constexpr const char* kDefaultTransportLibrary = "libasu_transport.so";

std::string GetEnvValue(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') { return {}; }
    return value;
}

std::filesystem::path ExecutableDir()
{
    char buffer[PATH_MAX] = {};
    const auto size = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (size <= 0) { return {}; }
    buffer[size] = '\0';
    return std::filesystem::path(buffer).parent_path();
}

std::vector<std::string> BuildLibraryCandidates(const std::string& explicitPath,
                                                const char* envName, const char* libraryName)
{
    std::vector<std::string> candidates;
    if (!explicitPath.empty()) { candidates.emplace_back(explicitPath); }
    auto envValue = GetEnvValue(envName);
    if (!envValue.empty()) { candidates.emplace_back(std::move(envValue)); }
    candidates.emplace_back(libraryName);

    const auto exeDir = ExecutableDir();
    if (!exeDir.empty()) {
        candidates.emplace_back((exeDir / libraryName).string());
        candidates.emplace_back((exeDir / ".." / "asu" / libraryName).lexically_normal().string());
    }
    return candidates;
}

void* OpenLibrary(const std::vector<std::string>& candidates, int flags, std::string& error)
{
    for (const auto& path : candidates) {
        dlerror();
        void* handle = dlopen(path.c_str(), flags);
        if (handle != nullptr) { return handle; }
        const char* dlError = dlerror();
        if (dlError != nullptr) { error = std::string{dlError}; }
    }
    return nullptr;
}

template <typename T>
Status LoadSymbol(void* handle, const char* name, T& symbol)
{
    dlerror();
    auto* rawSymbol = dlsym(handle, name);
    const char* error = dlerror();
    if (error != nullptr || rawSymbol == nullptr) {
        return Status::Error(kExitInvalidArgument,
                             "failed to load ASU runtime symbol " + std::string{name} + ": " +
                                 (error == nullptr ? "symbol is null" : std::string{error}));
    }
    symbol = reinterpret_cast<T>(rawSymbol);
    return Status::Success();
}

}  // namespace

AsuRuntimeProxy& AsuRuntimeProxy::Instance()
{
    static AsuRuntimeProxy proxy;
    return proxy;
}

Status AsuRuntimeProxy::Load(const AsuRuntimeLibraryConfig& config)
{
    if (clientHandle_ != nullptr && transportHandle_ != nullptr && createClient_ != nullptr &&
        createTransport_ != nullptr && loadClientConfig_ != nullptr) {
        return Status::Success();
    }

    config_ = config;
    std::string error;
    transportHandle_ =
        OpenLibrary(BuildLibraryCandidates(config_.transportLibraryPath, kTransportLibraryEnv,
                                           kDefaultTransportLibrary),
                    RTLD_NOW | RTLD_GLOBAL, error);
    if (transportHandle_ == nullptr) {
        return Status::Error(kExitInvalidArgument,
                             "failed to load ASU transport library: " + error);
    }

    clientHandle_ = OpenLibrary(
        BuildLibraryCandidates(config_.clientLibraryPath, kClientLibraryEnv, kDefaultClientLibrary),
        RTLD_NOW | RTLD_GLOBAL, error);
    if (clientHandle_ == nullptr) {
        return Status::Error(kExitInvalidArgument, "failed to load ASU client library: " + error);
    }

    auto status = LoadSymbol(transportHandle_, "UcmAsuCreateAsuTransport", createTransport_);
    if (!status.Ok()) { return status; }
    status = LoadSymbol(clientHandle_, "UcmAsuCreateAsuClient", createClient_);
    if (!status.Ok()) { return status; }
    return LoadSymbol(clientHandle_, "UcmAsuLoadAsuClientConfig", loadClientConfig_);
}

Status AsuRuntimeProxy::EnsureLoaded()
{
    if (clientHandle_ != nullptr && transportHandle_ != nullptr && createClient_ != nullptr &&
        createTransport_ != nullptr && loadClientConfig_ != nullptr) {
        return Status::Success();
    }
    return Load(AsuRuntimeLibraryConfig{});
}

UC::ASU::Status AsuRuntimeProxy::LoadAsuClientConfig(const std::string& configPath,
                                                     UC::ASU::AsuClientConfig& config)
{
    auto status = EnsureLoaded();
    if (!status.Ok()) {
        return UC::ASU::Status::Error(UC::ASU::StatusCode::INVALID_ARGUMENT, status.message);
    }
    return loadClientConfig_(configPath.c_str(), &config);
}

std::unique_ptr<UC::ASU::AsuClient> AsuRuntimeProxy::CreateAsuClient(
    const UC::ASU::TransportFactory* transportFactory, Status& status)
{
    status = EnsureLoaded();
    if (!status.Ok()) { return nullptr; }
    auto client = createClient_(transportFactory);
    if (client == nullptr) {
        status = Status::Error(kExitInvalidArgument, "ASU runtime returned null client");
    }
    return client;
}

std::unique_ptr<UC::ASU::AsuTransport> AsuRuntimeProxy::CreateAsuTransport(Status& status)
{
    status = EnsureLoaded();
    if (!status.Ok()) { return nullptr; }
    auto transport = createTransport_();
    if (transport == nullptr) {
        status = Status::Error(kExitInvalidArgument, "ASU runtime returned null transport");
    }
    return transport;
}

}  // namespace UC::KVTest
