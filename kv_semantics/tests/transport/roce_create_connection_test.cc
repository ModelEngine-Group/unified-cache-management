#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "roce/cpu_kv_over_roce_trans_provider.h"
#include "roce_test_support.h"

namespace kv::test {
namespace {

using ConnectionHandles = std::vector<TransProvider::ConnectionHandle>;

TEST(CpuKvOverRoceCreateConnectionTest, RunsTheHandshakeOnBothSides)
{
    const std::uint16_t port = ReserveTcpPort();
    ASSERT_NE(0, port);

    roce::CpuKvOverRoceTransProvider server(MakeServerConfig());

    Status serverStatus = Status::Error(StatusCode::INTERNAL_ERROR, "server thread did not run");
    ConnectionHandles serverHandles;
    std::thread serverThread([&]() {
        serverStatus =
            server.CreateConnection("127.0.0.1", "127.0.0.1", port, 2, 4000, serverHandles);
    });

    roce::CpuKvOverRoceTransProvider client(MakeClientConfig());
    ConnectionHandles clientHandles;
    const Status clientStatus = ConnectWithRetry(client, port, 2, clientHandles);

    serverThread.join();

    ASSERT_TRUE(clientStatus.ok()) << clientStatus.message;
    ASSERT_TRUE(serverStatus.ok()) << serverStatus.message;

    // One handle per requested queue pair, on both ends.
    EXPECT_EQ(2U, clientHandles.size());
    EXPECT_EQ(2U, serverHandles.size());
    EXPECT_EQ(1U, client.ConnectionCount());
    EXPECT_EQ(2U, client.HandleCount());
    EXPECT_EQ(1U, server.ConnectionCount());

    // The capability reply of the peer becomes the server capabilities.
    ServerKvCapabilities capabilities{};
    ASSERT_TRUE(client.GetServerCapabilities(clientHandles.front(), capabilities).ok());
    EXPECT_EQ(2U, capabilities.queueNum);
    EXPECT_EQ(8U, capabilities.keyLength);

    const std::vector<Status> clientDeleted = client.DeleteConnections(clientHandles);
    ASSERT_EQ(clientHandles.size(), clientDeleted.size());
    for (const Status& status : clientDeleted) { EXPECT_TRUE(status.ok()) << status.message; }
    EXPECT_EQ(0U, client.ConnectionCount());
    EXPECT_EQ(0U, client.HandleCount());

    const std::vector<Status> serverDeleted = server.DeleteConnections(serverHandles);
    for (const Status& status : serverDeleted) { EXPECT_TRUE(status.ok()) << status.message; }
    EXPECT_EQ(0U, server.ConnectionCount());
}

TEST(CpuKvOverRoceCreateConnectionTest, AdoptsTheQueuePairCountOfTheStore)
{
    const std::uint16_t port = ReserveTcpPort();
    ASSERT_NE(0, port);

    roce::CpuKvOverRoceTransProvider server(MakeServerConfig());
    Status serverStatus = Status::Error(StatusCode::INTERNAL_ERROR, "server thread did not run");
    ConnectionHandles serverHandles;
    std::thread serverThread([&]() {
        serverStatus =
            server.CreateConnection("127.0.0.1", "127.0.0.1", port, 1, 4000, serverHandles);
    });

    roce::CpuKvOverRoceTransProvider client(MakeClientConfig());
    ConnectionHandles clientHandles;
    const Status clientStatus = ConnectWithRetry(client, port, 4, clientHandles);

    serverThread.join();

    // The store announced a single queue pair, so the client links with one
    // instead of failing on the mismatch.
    ASSERT_TRUE(clientStatus.ok()) << clientStatus.message;
    ASSERT_TRUE(serverStatus.ok()) << serverStatus.message;
    EXPECT_EQ(1U, clientHandles.size());
    EXPECT_EQ(1U, serverHandles.size());

    ServerKvCapabilities capabilities;
    ASSERT_TRUE(client.GetServerCapabilities(clientHandles.front(), capabilities).ok());
    EXPECT_EQ(1U, capabilities.queueNum);

    EXPECT_TRUE(client.DeleteConnections(clientHandles).front().ok());
    EXPECT_TRUE(server.DeleteConnections(serverHandles).front().ok());
    EXPECT_EQ(0U, client.ConnectionCount());
    EXPECT_EQ(0U, server.ConnectionCount());
}

TEST(CpuKvOverRoceCreateConnectionTest, RejectsInvalidArguments)
{
    roce::CpuKvOverRoceTransProvider provider(MakeClientConfig());

    ConnectionHandles handles;

    Status status = provider.CreateConnection("127.0.0.1", "127.0.0.1", 19001, 0, 500, handles);
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, status.code);
    EXPECT_TRUE(handles.empty());

    status = provider.CreateConnection("127.0.0.1", "", 19001, 1, 500, handles);
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, status.code);
    EXPECT_TRUE(handles.empty());

    status = provider.CreateConnection("127.0.0.1", "127.0.0.1", 0, 1, 500, handles);
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, status.code);
    EXPECT_TRUE(handles.empty());

    // The out-of-band role is a configuration knob, so a typo has to be caught.
    TransportConfig badRole = MakeClientConfig();
    badRole.attrs["roce_oob_role"] = "peer";
    roce::CpuKvOverRoceTransProvider typoProvider(badRole);
    status = typoProvider.CreateConnection("127.0.0.1", "127.0.0.1", 19001, 1, 500, handles);
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, status.code);
    EXPECT_NE(std::string::npos, status.message.find("roce_oob_role"));
    EXPECT_TRUE(handles.empty());
    EXPECT_EQ(0U, provider.ConnectionCount());
}

TEST(CpuKvOverRoceCreateConnectionTest, RejectsAnUnknownQueuePairBackend)
{
    TransportConfig config = MakeClientConfig();
    config.attrs["roce_qp_backend"] = "magic";

    roce::CpuKvOverRoceTransProvider provider(config);
    ConnectionHandles handles;
    const Status status =
        provider.CreateConnection("127.0.0.1", "127.0.0.1", 19001, 1, 500, handles);

    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, status.code);
    EXPECT_NE(std::string::npos, status.message.find("roce_qp_backend"));
    EXPECT_TRUE(handles.empty());
}

TEST(CpuKvOverRoceCreateConnectionTest, RejectsKnobsThatDoNotFitTheirField)
{
    // Every one of these would silently wrap on the way into the verbs
    // attributes and the protocol frame, so the request is refused before a
    // socket is opened.  The last value is the interesting one: 2^32 used to be
    // truncated to zero first and then accepted as "not set".
    const struct {
        const char* key;
        const char* value;
    } cases[] = {
        {"roce_sl",               "300"       },
        {"roce_qp_timeout",       "300"       },
        {"roce_sl",               "4294967296"},
        {"roce_inline_threshold", "4294967296"},
    };
    {
        for (const auto& testCase : cases) {
            const char* knob = testCase.key;
            const char* value = testCase.value;
            TransportConfig config = MakeClientConfig();
            config.attrs[knob] = value;

            roce::CpuKvOverRoceTransProvider provider(config);
            ConnectionHandles handles;
            const Status status =
                provider.CreateConnection("127.0.0.1", "127.0.0.1", 19001, 1, 500, handles);

            EXPECT_EQ(StatusCode::INVALID_ARGUMENT, status.code) << knob << "=" << value;
            EXPECT_NE(std::string::npos, status.message.find(knob)) << status.message;
            EXPECT_NE(std::string::npos, status.message.find("above the maximum"))
                << status.message;
            EXPECT_TRUE(handles.empty()) << knob;
        }
    }

    // A port number starts at one, so zero is not a port.
    for (const auto& value : {"0", "256"}) {
        TransportConfig config = MakeClientConfig();
        config.attrs["roce_hca_port"] = value;

        roce::CpuKvOverRoceTransProvider provider(config);
        ConnectionHandles handles;
        const Status status =
            provider.CreateConnection("127.0.0.1", "127.0.0.1", 19001, 1, 500, handles);

        EXPECT_EQ(StatusCode::INVALID_ARGUMENT, status.code) << value;
        EXPECT_NE(std::string::npos, status.message.find("roce_hca_port")) << status.message;
        EXPECT_TRUE(handles.empty());
    }

    // A port that does not parse is a typo, not an invitation to keep the default.
    for (const auto& value : {"abc", "12ab", "-1"}) {
        TransportConfig config = MakeClientConfig();
        config.attrs["roce_hca_port"] = value;

        roce::CpuKvOverRoceTransProvider provider(config);
        ConnectionHandles handles;
        const Status status =
            provider.CreateConnection("127.0.0.1", "127.0.0.1", 19001, 1, 500, handles);

        EXPECT_EQ(StatusCode::INVALID_ARGUMENT, status.code) << value;
        EXPECT_NE(std::string::npos, status.message.find("is not a number")) << status.message;
        EXPECT_TRUE(handles.empty()) << value;
    }
}

TEST(CpuKvOverRoceCreateConnectionTest, RejectsCapabilityLimitsThatDoNotFitTheMessage)
{
    // These limits are advertised in 16 bit fields, so a bigger value used to be
    // truncated on the way into the capability message and then negotiated as if
    // it were the configured one.
    const auto expectRejected = [](const char* name, const TransportConfig& config) {
        roce::CpuKvOverRoceTransProvider provider(config);
        ConnectionHandles handles;
        const Status status =
            provider.CreateConnection("127.0.0.1", "127.0.0.1", 19001, 1, 500, handles);

        EXPECT_EQ(StatusCode::INVALID_ARGUMENT, status.code) << name;
        EXPECT_NE(std::string::npos, status.message.find(name)) << status.message;
        EXPECT_NE(std::string::npos, status.message.find("capability field")) << status.message;
        EXPECT_TRUE(handles.empty()) << name;
    };

    {
        TransportConfig config = MakeClientConfig();
        config.maxInflightTasks = 0x10000U;
        expectRejected("max_inflight_tasks", config);
    }
    {
        TransportConfig config = MakeClientConfig();
        config.batchStoreIoNum = 0x10000U;
        expectRejected("asu_batch_store_io_num", config);
    }
    {
        TransportConfig config = MakeClientConfig();
        config.batchLoadIoNum = 0x10000U;
        expectRejected("asu_batch_load_io_num", config);
    }
    {
        TransportConfig config = MakeClientConfig();
        config.queryIoNum = 0x10000U;
        expectRejected("asu_query_io_num", config);
    }
}

TEST(CpuKvOverRoceCreateConnectionTest, AcceptsZeroForKnobsWhereItIsMeant)
{
    // Zero is a legitimate setting, not a missing one: the request has to get
    // past validation and fail on the network instead of on the knob.
    TransportConfig config = MakeClientConfig();
    config.attrs["roce_inline_threshold"] = "0";
    config.attrs["roce_gid_index"] = "0";
    config.attrs["roce_sl"] = "0";

    roce::CpuKvOverRoceTransProvider provider(config);
    ConnectionHandles handles;
    const Status status = provider.CreateConnection("127.0.0.1", "127.0.0.1", 1, 1, 200, handles);

    EXPECT_NE(StatusCode::INVALID_ARGUMENT, status.code) << status.message;
    EXPECT_TRUE(handles.empty());
}

TEST(CpuKvOverRoceCreateConnectionTest, FailsCleanlyWhenNobodyListens)
{
    const std::uint16_t port = ReserveTcpPort();
    ASSERT_NE(0, port);

    roce::CpuKvOverRoceTransProvider provider(MakeClientConfig());
    ConnectionHandles handles(3, reinterpret_cast<TransProvider::ConnectionHandle>(1));

    const Status status =
        provider.CreateConnection("127.0.0.1", "127.0.0.1", port, 2, 500, handles);

    EXPECT_FALSE(status.ok());
    EXPECT_NE(StatusCode::OK, status.code);
    EXPECT_NE(std::string::npos, status.message.find("out-of-band TCP connect"));
    // The contract: the handle list is empty whenever the call fails.
    EXPECT_TRUE(handles.empty());
    EXPECT_EQ(0U, provider.ConnectionCount());
}

TEST(CpuKvOverRoceCreateConnectionTest, DeleteReportsUnknownHandles)
{
    roce::CpuKvOverRoceTransProvider provider(MakeClientConfig());

    const std::vector<TransProvider::ConnectionHandle> handles{
        reinterpret_cast<TransProvider::ConnectionHandle>(0x1234)};
    const std::vector<Status> results = provider.DeleteConnections(handles);

    ASSERT_EQ(1U, results.size());
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, results.front().code);
}

}  // namespace
}  // namespace kv::test
