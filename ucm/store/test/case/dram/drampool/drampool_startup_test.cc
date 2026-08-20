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
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#if defined(UCM_DRAMPOOL_RUNTIME_INTEGRATION_TESTS) && !defined(_WIN32)
#include <arpa/inet.h>
#include <chrono>
#include <csignal>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include "channels/tcp/tcp_message_channel.h"
#include "kv_protocol.h"
#include "logger/logger.h"
#endif
#include "drampool_config.h"
#include "drampool_daemon.h"
#include "drampool_server.h"

namespace UC::DramPool {
namespace {

std::filesystem::path RepositoryRuntimeConfigPath();

class ScopedCurrentPath {
public:
    explicit ScopedCurrentPath(const std::filesystem::path& path)
        : previous_(std::filesystem::current_path())
    {
        std::filesystem::current_path(path);
    }

    ~ScopedCurrentPath() { std::filesystem::current_path(previous_); }

private:
    std::filesystem::path previous_;
};

#if defined(UCM_DRAMPOOL_RUNTIME_INTEGRATION_TESTS)
std::uint16_t FindAvailableTcpPort()
{
#if !defined(_WIN32)
    const int socket = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socket < 0) { throw std::runtime_error("failed to create test socket"); }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    socklen_t addressLength = sizeof(address);
    if (::bind(socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::getsockname(socket, reinterpret_cast<sockaddr*>(&address), &addressLength) != 0) {
        ::close(socket);
        throw std::runtime_error("failed to reserve an available test port");
    }
    ::close(socket);
    return ntohs(address.sin_port);
#else
    static std::uint16_t nextPort = 19000;
    return nextPort++;
#endif
}

std::uint16_t FindDistinctTcpPort(std::uint16_t first, std::uint16_t second = 0)
{
    std::uint16_t port = 0;
    do {
        port = FindAvailableTcpPort();
    } while (port == first || port == second);
    return port;
}

DramPoolConfig MakeValidConfig()
{
    const char* argv[] = {
        "drampool",
        "--addr",
        "127.0.0.1:9000",
        "--nics",
        "mlx5_0",
        "mlx5_1",
        "--pool-size-gb",
        "1",
        "--kvcache-block-sizes",
        "4096",
        "8192",
        "--kvcache-block-proportions",
        "70",
        "30",
    };
    DramPoolConfig config;
    if (const auto status = ParseCommandLine(14, const_cast<char**>(argv), config);
        status.Failure()) {
        throw std::runtime_error(status.ToString());
    }

    if (const auto status = ParseYamlConfig(RepositoryRuntimeConfigPath().string(), config);
        status.Failure()) {
        throw std::runtime_error(status.ToString());
    }
    const auto originalLocalId = config.addr.ToString();
    config.addr.port = FindAvailableTcpPort();
    const auto oneSidedPort = FindDistinctTcpPort(config.addr.port);
    config.twoSidedToOneSided.erase(originalLocalId);
    config.twoSidedToOneSided.emplace(config.addr.ToString(),
                                      "127.0.0.1:" + std::to_string(oneSidedPort));
    return config;
}

class ScopedDramPoolConfig {
public:
    explicit ScopedDramPoolConfig(DramPoolConfig config) : previous_(g_config)
    {
        g_config = std::move(config);
    }

    ~ScopedDramPoolConfig() { g_config = std::move(previous_); }

    ScopedDramPoolConfig(const ScopedDramPoolConfig&) = delete;
    ScopedDramPoolConfig& operator=(const ScopedDramPoolConfig&) = delete;

private:
    DramPoolConfig previous_;
};

#if !defined(_WIN32)
class ScopedListeningPort {
public:
    explicit ScopedListeningPort(std::uint16_t port)
    {
        socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (socket_ < 0) { throw std::runtime_error("failed to create test socket"); }
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        if (::bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
            ::listen(socket_, 1) != 0) {
            ::close(socket_);
            throw std::runtime_error("failed to occupy test port");
        }
    }

    ~ScopedListeningPort() { ::close(socket_); }

private:
    int socket_{-1};
};

bool CanConnectTo(std::uint16_t port)
{
    const int socket = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socket < 0) { return false; }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    const bool connected =
        ::connect(socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0;
    ::close(socket);
    return connected;
}

template <typename Scenario>
int RunInIsolatedProcess(Scenario scenario)
{
    const auto child = ::fork();
    if (child < 0) { return -1; }
    if (child == 0) { ::_exit(scenario()); }
    int status = 0;
    if (::waitpid(child, &status, 0) != child || !WIFEXITED(status)) { return -1; }
    return WEXITSTATUS(status);
}

std::filesystem::path CreateDaemonRuntimeDirectory(std::uint16_t servicePort,
                                                   std::uint16_t oneSidedPort)
{
    const auto directory =
        std::filesystem::temp_directory_path() / ("drampool_daemon_" + std::to_string(::getpid()));
    std::filesystem::create_directories(directory / "examples");
    std::ifstream input(RepositoryRuntimeConfigPath());
    std::string yaml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto replacePort = [&yaml](const std::string& original, std::uint16_t replacement) {
        const auto replacementText = std::to_string(replacement);
        auto position = yaml.find(original);
        if (position == std::string::npos) {
            throw std::runtime_error("expected endpoint missing from test runtime YAML");
        }
        while (position != std::string::npos) {
            yaml.replace(position, original.size(), replacementText);
            position = yaml.find(original, position + replacementText.size());
        }
    };
    replacePort("4501", oneSidedPort);
    replacePort("9000", servicePort);
    std::ofstream output(directory / "examples" / "drampool.yaml");
    output << yaml;
    return directory;
}
#endif
#endif

std::filesystem::path RepositoryRuntimeConfigPath()
{
    auto directory = std::filesystem::absolute(__FILE__).parent_path();
    while (!directory.empty()) {
        const auto configPath = directory / "examples" / "drampool.yaml";
        if (std::filesystem::exists(configPath)) { return configPath; }
        if (directory == directory.root_path()) { break; }
        directory = directory.parent_path();
    }
    return "examples/drampool.yaml";
}

}  // namespace

TEST(DramPoolConfigTest, ParsesLaunchOptions)
{
    const char* argv[] = {
        "drampool",
        "--addr",
        "127.0.0.1:9000",
        "--nics",
        "mlx5_0",
        "mlx5_1",
        "--pool-size-gb",
        "128",
        "--kvcache-block-sizes",
        "4096",
        "8192",
        "--kvcache-block-proportions",
        "1",
        "3",
        "--ttl-minutes=30",
        "--config=/tmp/drampool-custom.yaml",
    };
    DramPoolConfig config;

    const auto status = ParseCommandLine(16, const_cast<char**>(argv), config);

    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(config.addr.host, "127.0.0.1");
    EXPECT_EQ(config.addr.port, 9000U);
    EXPECT_EQ(config.nics, (std::vector<std::string>{"mlx5_0", "mlx5_1"}));
    EXPECT_EQ(config.poolSizeGb, 128U);
    EXPECT_EQ(config.poolBlockSizes, (std::vector<std::uint64_t>{4096, 8192}));
    EXPECT_EQ(config.poolBlockProportions, (std::vector<std::uint32_t>{1, 3}));
    EXPECT_EQ(config.poolSlotCounts, (std::vector<std::uint32_t>{8'388'608, 12'582'912}));
    EXPECT_EQ(config.defaultDumpTtlMs, 30U * kMillisecondsPerMinute);
    EXPECT_EQ(config.runtimeConfigPath, "/tmp/drampool-custom.yaml");
}

TEST(DramPoolConfigTest, DefaultsBlockProportionsAndTtl)
{
    const char* argv[] = {
        "drampool",          "--addr=127.0.0.1:9000",      "--nics=mlx5_0",
        "--pool-size-gb=64", "--kvcache-block-sizes=4096", "8192",
    };
    DramPoolConfig config;

    const auto status = ParseCommandLine(6, const_cast<char**>(argv), config);

    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(config.poolBlockProportions, (std::vector<std::uint32_t>{1, 1}));
    EXPECT_EQ(config.defaultDumpTtlMs, kDefaultDumpTtlMs);
    EXPECT_EQ(config.runtimeConfigPath, kDefaultDramPoolRuntimeConfigPath);
    EXPECT_NE(BuildUsage("drampool").find("--config <PATH>"), std::string::npos);
}

TEST(DramPoolConfigTest, ResolvesGiBSlotCountsWithAlignedStride)
{
    const char* argv[] = {
        "drampool",         "--addr=127.0.0.1:9000",      "--nics=mlx5_0",
        "--pool-size-gb=1", "--kvcache-block-sizes=4097",
    };
    DramPoolConfig config;

    const auto status = ParseCommandLine(5, const_cast<char**>(argv), config);

    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(config.poolSlotCounts, (std::vector<std::uint32_t>{258'111}));
}

TEST(DramPoolRuntimeConfigTest, LoadsRepositoryExample)
{
    const char* argv[] = {
        "drampool",       "--addr", "127.0.0.1:9000",        "--nics", "mlx5_0",
        "--pool-size-gb", "1",      "--kvcache-block-sizes", "4096",
    };
    DramPoolConfig config;
    ASSERT_TRUE(ParseCommandLine(9, const_cast<char**>(argv), config).Success());

    const auto status = ParseYamlConfig(RepositoryRuntimeConfigPath().string(), config);

    ASSERT_TRUE(status.Success()) << status.ToString();
    ASSERT_EQ(config.twoSidedToOneSided.size(), 2U);
    EXPECT_EQ(config.twoSidedToOneSided.at("127.0.0.1:9000"), "127.0.0.1:4501");
    EXPECT_EQ(config.requestQueueDepth, 65536U);
    EXPECT_EQ(config.completionQueueDepth, 65536U);
    EXPECT_EQ(config.requestReceiverIdleWaitUs, 100U);
    EXPECT_EQ(config.pollerPendingDepth, 64U);
    EXPECT_EQ(config.metadataPeriodicEvictionPolicy, EvictionPolicyType::TTL);
    EXPECT_EQ(config.metadataDeepEvictionPolicy, EvictionPolicyType::POSITION);
    EXPECT_EQ(config.metadataLeaseTimeMs, 5000U);
    EXPECT_DOUBLE_EQ(config.metadataDefaultEvictRatio, 0.0);
    EXPECT_EQ(config.metadataEvictPeriodMs, 31'536'000'000ULL);
}

TEST(DramPoolConfigTest, RejectsInvalidCommandLines)
{
    {
        const char* argv[] = {"drampool", "--help"};
        DramPoolConfig config;
        EXPECT_TRUE(ParseCommandLine(2, const_cast<char**>(argv), config).Failure());
    }
    {
        const char* argv[] = {"drampool"};
        DramPoolConfig config;
        EXPECT_TRUE(ParseCommandLine(1, const_cast<char**>(argv), config).Failure());
    }
    {
        const char* argv[] = {
            "drampool", "--addr", "127.0.0.1:9000", "--pool-size-gb", "1", "--kvcache-block-sizes",
            "4096"};
        DramPoolConfig config;
        EXPECT_TRUE(ParseCommandLine(7, const_cast<char**>(argv), config).Failure());
    }
    {
        const char* argv[] = {"drampool", "--config"};
        DramPoolConfig config;
        EXPECT_TRUE(ParseCommandLine(2, const_cast<char**>(argv), config).Failure());
    }
    {
        const char* argv[] = {
            "drampool",
            "--addr=127.0.0.1:9000",
            "--nics=mlx5_0",
            "--pool-size-gb=1",
            "--kvcache-block-sizes=4096",
            "--config=first.yaml",
            "--config=second.yaml",
        };
        DramPoolConfig config;
        const auto status = ParseCommandLine(7, const_cast<char**>(argv), config);
        EXPECT_TRUE(status.Failure());
        EXPECT_NE(status.ToString().find("--config may be specified once"), std::string::npos);
    }
    {
        const char* argv[] = {
            "drampool",         "--addr=127.0.0.1:9000",      "--nics=mlx5_0",
            "--pool-size-gb=1", "--kvcache-block-sizes=4096", "--config=   ",
        };
        DramPoolConfig config;
        const auto status = ParseCommandLine(6, const_cast<char**>(argv), config);
        EXPECT_TRUE(status.Failure());
        EXPECT_NE(status.ToString().find("--config must not be blank"), std::string::npos);
    }
    {
        const char* argv[] = {
            "drampool",
            "--addr",
            "127.0.0.1:9000",
            "--nics",
            "mlx5_0",
            "--pool-size-gb",
            "1",
            "--kvcache-block-sizes",
            "4096",
            "8192",
            "--kvcache-block-proportions",
            "1",
            "--ttl-minutes",
            "0",
        };
        DramPoolConfig config;
        EXPECT_TRUE(ParseCommandLine(14, const_cast<char**>(argv), config).Failure());
    }
    {
        const char* argv[] = {
            "drampool",
            "--addr",
            "127.0.0.1:9000",
            "--nics",
            "mlx5_0",
            "--pool-size-gb=-1",
            "--kvcache-block-sizes",
            "4096",
        };
        DramPoolConfig config;
        EXPECT_TRUE(ParseCommandLine(8, const_cast<char**>(argv), config).Failure());
    }
    {
        const char* argv[] = {"drampool", "--pool-size-gb", "0"};
        DramPoolConfig config;
        EXPECT_TRUE(ParseCommandLine(3, const_cast<char**>(argv), config).Failure());
    }
    {
        const char* argv[] = {"drampool", "--kvcache-block-sizes", "0"};
        DramPoolConfig config;
        EXPECT_TRUE(ParseCommandLine(3, const_cast<char**>(argv), config).Failure());
    }
    {
        const char* argv[] = {"drampool", "--kvcache-block-proportions", "0"};
        DramPoolConfig config;
        EXPECT_TRUE(ParseCommandLine(3, const_cast<char**>(argv), config).Failure());
    }
    {
        const char* argv[] = {
            "drampool",
            "--pool-size-gb",
            "18446744073709551615",
            "--unknown",
        };
        DramPoolConfig config;
        const auto status = ParseCommandLine(4, const_cast<char**>(argv), config);
        EXPECT_TRUE(status.Failure());
        EXPECT_NE(status.ToString().find("--pool-size-gb is too large"), std::string::npos);
    }
    {
        const char* argv[] = {
            "drampool",
            "--addr",
            "127.0.0.1:9000",
            "--nics",
            "mlx5_0",
            "--pool-size-gb",
            "1",
            "--kvcache-block-proportions",
            "1",
            "--kvcache-block-sizes",
            "4096",
            "8192",
        };
        DramPoolConfig config;
        const auto status = ParseCommandLine(12, const_cast<char**>(argv), config);
        EXPECT_TRUE(status.Failure());
        EXPECT_NE(status.ToString().find("must have the same length"), std::string::npos);
    }
    {
        const char* argv[] = {
            "drampool",       "--addr", "127.0.0.1:bad",         "--nics", "mlx5_0",
            "--pool-size-gb", "1",      "--kvcache-block-sizes", "4096",
        };
        DramPoolConfig config;
        EXPECT_TRUE(ParseCommandLine(9, const_cast<char**>(argv), config).Failure());
    }
}

TEST(DramPoolServerTest, RejectsCallsOutsideValidState)
{
    DramPoolServer server;
    EXPECT_TRUE(server.Start().Failure());
    server.Stop();
    server.Stop();
}

#if defined(UCM_DRAMPOOL_RUNTIME_INTEGRATION_TESTS)
#if !defined(_WIN32)
TEST(DramPoolServerTest, RequestReceiverLogsReceivedRequestFields)
{
    constexpr std::uint64_t kRequestId = 42;
    const auto logRoot = std::filesystem::temp_directory_path() /
                         ("drampool_request_receiver_log_" + std::to_string(::getpid()));
    std::filesystem::remove_all(logRoot);
    std::filesystem::create_directories(logRoot);

    const auto child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        (void)::setenv("UC_LOGGER_LEVEL", "info", 1);
        (void)::setenv("UCM_LOG_RATE_LIMIT_ENABLE", "false", 1);
        UC::Logger::Setup(logRoot.string(), 1, 1);

        auto config = MakeValidConfig();
        config.poolBlockSizes = {4096};
        config.poolBlockProportions = {1};
        config.poolSlotCounts = {1};
        config.gcEnabled = false;
        const transport::Endpoint clientControl{"127.0.0.1", FindDistinctTcpPort(config.addr.port)};
        const auto clientOneSidedPort = FindDistinctTcpPort(config.addr.port, clientControl.port);
        config.twoSidedToOneSided.emplace(clientControl.ToString(),
                                          "127.0.0.1:" + std::to_string(clientOneSidedPort));
        ScopedDramPoolConfig configScope(std::move(config));

        DramPoolServer server;
        transport::TcpMessageChannel client;
        ProtocolManager protocol;
        if (server.Init().Failure()) { ::_exit(1); }
        if (server.Start().Failure()) { ::_exit(2); }
        if (client.Init(clientControl).Failure()) { ::_exit(3); }

        KvLookupRequest request;
        request.opcode = OpType::LOOKUP;
        request.request_id = kRequestId;
        request.resp_addr = 0x1000;
        request.batch_size = 1;
        request.entries.emplace_back();
        request.entries.back().key.back() = std::byte{1};
        const auto packedSize = protocol.GetPackedRequestSize(request.opcode, request);
        std::vector<std::uint8_t> packed(packedSize);
        if (protocol.PackRequest(packed.data(), request.opcode, request).Failure()) { ::_exit(4); }
        if (client.Send(g_config.addr, packed.data(), packed.size()).Failure()) { ::_exit(5); }

        const auto expected =
            "RequestReceiver received request, request_id=" + std::to_string(kRequestId) +
            ", opcode=" + std::to_string(static_cast<int>(OpType::LOOKUP));
        const auto logPath = logRoot / std::to_string(::getpid()) / "ucm.log";
        bool stageFound = false;
        bool completionFound = false;
        for (int attempt = 0; attempt < 200 && (!stageFound || !completionFound); ++attempt) {
            UC::Logger::Flush();
            std::ifstream input(logPath);
            const std::string content((std::istreambuf_iterator<char>(input)),
                                      std::istreambuf_iterator<char>());
            stageFound = content.find(expected) != std::string::npos;
            completionFound = content.find(completed) != std::string::npos;
            if (!stageFound || !completionFound) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        (void)client.Shutdown();
        server.Stop();
        UC::Logger::Flush();
        ::_exit(stageFound && completionFound ? 0 : 6);
    }

    int childStatus = 0;
    ASSERT_EQ(::waitpid(child, &childStatus, 0), child);
    ASSERT_TRUE(WIFEXITED(childStatus));
    ASSERT_EQ(WEXITSTATUS(childStatus), 0);

    const auto logPath = logRoot / std::to_string(child) / "ucm.log";
    std::ifstream input(logPath);
    const std::string content((std::istreambuf_iterator<char>(input)),
                              std::istreambuf_iterator<char>());
    const auto messagePosition = content.find("[PERF] component=drampool event=stage request_id=");
    ASSERT_NE(messagePosition, std::string::npos);
    const auto messageEnd = content.find('\n', messagePosition);
    std::cout << content.substr(messagePosition, messageEnd - messagePosition) << std::endl;
    const auto completionPosition =
        content.find("[PERF] component=drampool event=request_done request_id=");
    ASSERT_NE(completionPosition, std::string::npos);
    const auto completionEnd = content.find('\n', completionPosition);
    std::cout << content.substr(completionPosition, completionEnd - completionPosition)
              << std::endl;
    std::filesystem::remove_all(logRoot);
}

TEST(DramPoolServerTest, StartsAndStopsService)
{
    EXPECT_EQ(RunInIsolatedProcess([]() {
                  ScopedDramPoolConfig configScope(MakeValidConfig());
                  DramPoolServer server;
                  if (server.Init().Failure()) { return 1; }
                  if (server.Init().Success()) { return 2; }
                  if (server.Start().Failure()) { return 3; }
                  if (server.Start().Success()) { return 4; }
                  server.Stop();
                  server.Stop();
                  if (server.Init().Success()) { return 5; }
                  if (server.Start().Success()) { return 6; }
                  return 0;
              }),
              0);
}

TEST(DramPoolServerTest, RejectsDuplicateInitializationAndCleansUpWithoutStart)
{
    EXPECT_EQ(RunInIsolatedProcess([]() {
                  ScopedDramPoolConfig configScope(MakeValidConfig());
                  DramPoolServer server;
                  if (server.Init().Failure()) { return 1; }
                  return server.Init().Failure() ? 0 : 2;
              }),
              0);
}

TEST(DramPoolServerTest, StartFailureRollsBackStartedComponents)
{
    EXPECT_EQ(RunInIsolatedProcess([]() {
                  auto config = MakeValidConfig();
                  ScopedListeningPort occupiedServicePort(config.addr.port);
                  ScopedDramPoolConfig configScope(std::move(config));
                  DramPoolServer server;
                  if (server.Init().Failure()) { return 1; }
                  if (server.Start().Success()) { return 2; }
                  server.Stop();
                  return server.Start().Failure() ? 0 : 3;
              }),
              0);
}

TEST(DramPoolServerTest, StartsAndStopsWithGcDisabled)
{
    EXPECT_EQ(RunInIsolatedProcess([]() {
                  auto config = MakeValidConfig();
                  config.gcEnabled = false;
                  config.gcIntervalMs = 0;
                  ScopedDramPoolConfig configScope(std::move(config));
                  DramPoolServer server;
                  if (server.Init().Failure()) { return 1; }
                  if (server.Start().Failure()) { return 2; }
                  server.Stop();
                  return 0;
              }),
              0);
}
#endif
#endif

TEST(DramPoolDaemonTest, ReturnsForInvalidArguments)
{
    const std::vector<std::vector<const char*>> cases = {
        {"drampool"},
        {"drampool", "--help"},
        {"drampool", "--config"},
    };
    for (const auto& arguments : cases) {
        std::vector<char*> argv;
        argv.reserve(arguments.size());
        for (const auto* argument : arguments) { argv.push_back(const_cast<char*>(argument)); }
        DramPoolDaemon daemon;
        EXPECT_EQ(daemon.Run(static_cast<int>(argv.size()), argv.data()), 1);
    }
}

TEST(DramPoolDaemonTest, ReturnsWhenRuntimeYamlIsMissing)
{
    const auto emptyDirectory = std::filesystem::temp_directory_path() / "drampool_no_yaml";
    std::filesystem::create_directories(emptyDirectory);
    {
        ScopedCurrentPath pathScope(emptyDirectory);
        const char* argv[] = {
            "drampool",       "--addr", "127.0.0.1:19000",       "--nics", "mlx5_0",
            "--pool-size-gb", "1",      "--kvcache-block-sizes", "4096",
        };
        DramPoolDaemon daemon;

        EXPECT_EQ(daemon.Run(9, const_cast<char**>(argv)), 1);
    }
    std::filesystem::remove(emptyDirectory);
}

TEST(DramPoolDaemonTest, ReturnsWhenConfiguredRuntimeYamlIsMissing)
{
    const auto missingPath =
        (std::filesystem::temp_directory_path() / "drampool_missing_explicit.yaml").string();
    std::vector<std::string> arguments = {
        "drampool",       "--addr", "127.0.0.1:19000",       "--nics", "mlx5_0",
        "--pool-size-gb", "1",      "--kvcache-block-sizes", "4096",   "--config",
        missingPath,
    };
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) { argv.push_back(argument.data()); }
    DramPoolDaemon daemon;

    EXPECT_EQ(daemon.Run(static_cast<int>(argv.size()), argv.data()), 1);
    EXPECT_EQ(g_config.runtimeConfigPath, missingPath);
}

#if defined(UCM_DRAMPOOL_RUNTIME_INTEGRATION_TESTS) && !defined(_WIN32)
TEST(DramPoolDaemonTest, RunsUntilSigtermAndShutsDownCleanly)
{
    const auto servicePort = FindAvailableTcpPort();
    const auto oneSidedPort = FindDistinctTcpPort(servicePort);
    const auto runtimeDirectory = CreateDaemonRuntimeDirectory(servicePort, oneSidedPort);
    const auto runtimeConfigPath = (runtimeDirectory / "examples" / "drampool.yaml").string();
    const auto child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        std::filesystem::current_path(runtimeDirectory);
        const auto serviceEndpoint = "127.0.0.1:" + std::to_string(servicePort);
        std::vector<std::string> arguments = {
            "drampool",        "--addr", serviceEndpoint,         "--nics", "mlx5_0",
            "--pool-size-gb",  "1",      "--kvcache-block-sizes", "4096",   "--config",
            runtimeConfigPath,
        };
        std::vector<char*> argv;
        argv.reserve(arguments.size());
        for (auto& argument : arguments) { argv.push_back(argument.data()); }
        DramPoolDaemon daemon;
        const int result = daemon.Run(static_cast<int>(argv.size()), argv.data());
        ::_exit(result);
    }

    bool ready = false;
    bool exitedEarly = false;
    int childStatus = 0;
    for (int attempt = 0; attempt < 200; ++attempt) {
        if (::waitpid(child, &childStatus, WNOHANG) == child) {
            exitedEarly = true;
            break;
        }
        if (CanConnectTo(servicePort)) {
            ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!exitedEarly) {
        EXPECT_EQ(::kill(child, SIGTERM), 0);
        EXPECT_EQ(::waitpid(child, &childStatus, 0), child);
    }
    std::filesystem::remove_all(runtimeDirectory);
    ASSERT_TRUE(ready) << "daemon did not open its service listener";
    ASSERT_TRUE(WIFEXITED(childStatus));
    EXPECT_EQ(WEXITSTATUS(childStatus), 0);
}
#endif

}  // namespace UC::DramPool
