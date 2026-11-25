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

// Provider level tests of the CPU data path.  The send/response wire format
// itself is covered by roce_io_wire_test.cc; the cases here link a real provider
// pair over loopback and check what Send() reports without RDMA hardware.

#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <vector>
#include "kv_types.h"
#include "roce/cpu_kv_over_roce_trans_provider.h"
#include "roce/roce_io_wire.h"
#include "roce_test_support.h"

namespace kv::test {
namespace {

// The simulated backend has no completion queue, so the CPU data path cannot run
// on it; the provider has to say so per batch instead of failing the whole call.
TEST(CpuKvOverRoceSendTest, SimulatedBackendReportsSendUnsupported)
{
    const std::uint16_t port = ReserveTcpPort();
    ASSERT_NE(0, port);

    roce::CpuKvOverRoceTransProvider server(MakeServerConfig());
    Status serverStatus = Status::Error(StatusCode::INTERNAL_ERROR, "server thread did not run");
    std::vector<TransProvider::ConnectionHandle> serverHandles;
    std::thread serverThread([&]() {
        serverStatus =
            server.CreateConnection("127.0.0.1", "127.0.0.1", port, 1, 4000, serverHandles);
    });
    ThreadJoiner serverJoiner(serverThread);

    roce::CpuKvOverRoceTransProvider client(MakeClientConfig());
    std::vector<TransProvider::ConnectionHandle> clientHandles;
    const Status clientStatus = ConnectWithRetry(client, port, 1, clientHandles);
    serverThread.join();

    ASSERT_TRUE(clientStatus.ok()) << clientStatus.message;
    ASSERT_TRUE(serverStatus.ok()) << serverStatus.message;
    ASSERT_EQ(1U, clientHandles.size());

    std::uint32_t sqe[roce::kSqeHeaderDwords] = {};
    std::uint32_t flag[8] = {};
    const std::vector<TransProvider::SendIoBatch> batches{
        {clientHandles.front(), sqe, flag, sizeof(sqe)}
    };
    const std::vector<Status> sent = client.Send(batches, 1, 1);
    ASSERT_EQ(batches.size(), sent.size());
    EXPECT_EQ(StatusCode::UNSUPPORTED, sent.front().code);
    EXPECT_NE(std::string::npos, sent.front().message.find("roce_qp_backend"))
        << sent.front().message;

    const std::vector<Status> clientDeleted = client.DeleteConnections(clientHandles);
    for (const Status& status : clientDeleted) { EXPECT_TRUE(status.ok()) << status.message; }
    const std::vector<Status> serverDeleted = server.DeleteConnections(serverHandles);
    for (const Status& status : serverDeleted) { EXPECT_TRUE(status.ok()) << status.message; }
}

}  // namespace
}  // namespace kv::test
