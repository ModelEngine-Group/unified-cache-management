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
#include "drampool_daemon.h"
#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>
#include "drampool_config.h"
#include "drampool_server.h"
#include "health_server.h"
#include "logger/logger.h"

namespace UC::DramPool {
int DramPoolDaemon::Run(int argc, char** argv)
{
    auto status = ParseCommandLine(argc, argv, g_config);
    if (status.Failure()) {
        std::cerr << status.ToString() << "\n" << BuildUsage(argc > 0 ? argv[0] : "drampool");
        return 1;
    }
    status = ParseYamlConfig(g_config.runtimeConfigPath, g_config);
    if (status.Failure()) {
        std::cerr << status.ToString() << "\n";
        return 1;
    }
    status = SetupLogger();
    if (status.Failure()) {
        std::cerr << status.ToString() << "\n";
        return 1;
    }
    status = SetupSignals();
    if (status.Failure()) {
        UC_ERROR_UNLIMITED("DramPool setup signals failed: {}", status);
        return 1;
    }

    DramPoolServer server;
    HealthServer healthServer;
    status = server.Init();
    if (status.Failure()) {
        UC_ERROR_UNLIMITED("DramPool server init failed: {}", status);
        return 1;
    }
    status = server.Start();
    if (status.Failure()) {
        UC_ERROR_UNLIMITED("DramPool server start failed: {}", status);
        return 1;
    }

    status = healthServer.Start();
    if (status.Failure()) {
        UC_ERROR_UNLIMITED("DramPool health server start failed: {}", status);
        server.Stop();
        return 1;
    }

    UC_INFO_UNLIMITED("DramPool service ready, addr={}", g_config.addr.ToString());
    WaitForShutdown();
    UC_INFO_UNLIMITED("DramPool shutdown requested");
    healthServer.Stop();
    server.Stop();
    UC::Logger::Flush();
    return 0;
}

Status DramPoolDaemon::SetupLogger()
{
    UC::Logger::Setup(g_config.logDir, static_cast<int>(g_config.logMaxFiles),
                      static_cast<int>(g_config.logMaxSizeMb));
    return Status::OK();
}

Status DramPoolDaemon::SetupSignals()
{
    shutdownRequested_.store(false, std::memory_order_release);
    if (std::signal(SIGINT, &DramPoolDaemon::HandleSignal) == SIG_ERR) {
        return Status::OsApiError("failed to install SIGINT handler");
    }
    if (std::signal(SIGTERM, &DramPoolDaemon::HandleSignal) == SIG_ERR) {
        return Status::OsApiError("failed to install SIGTERM handler");
    }
    return Status::OK();
}

void DramPoolDaemon::WaitForShutdown()
{
    constexpr auto kShutdownPollInterval = std::chrono::milliseconds(100);

    while (!shutdownRequested_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(kShutdownPollInterval);
    }
}

void DramPoolDaemon::HandleSignal(int signum)
{
    if (signum == SIGINT || signum == SIGTERM) {
        shutdownRequested_.store(true, std::memory_order_release);
    }
}

}  // namespace UC::DramPool
