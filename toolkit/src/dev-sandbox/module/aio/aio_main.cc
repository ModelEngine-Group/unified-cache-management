/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#include <chrono>
#include <random>
#include "aio_engine.h"
#include "host_buffer.h"

struct Config {
    std::string workspace;
    aio::HostBuffer::Strategy ioType = aio::HostBuffer::Strategy::MMAP;
    size_t ioSize = 1024 * 1024;
    size_t ioNumber = 512;
    size_t deviceId = 0;
    size_t epochNumber = 32;

    bool Parse(int argc, char const* argv[])
    {
        auto PrintUsage = +[](const char* prog) {
            fmt::println(
                "Usage: {} --workspace <path> [--io-type mmap|alloc] [--io-size <bytes>] "
                "[--io-number <n>] [--device-id <id>] [--epoch-number <n>]",
                prog ? prog : "aio_main");
        };
        for (int i = 1; i < argc; ++i) {
            std::string opt = argv[i];
            if (opt == "-h" || opt == "--help") {
                PrintUsage(argv[0]);
                return false;
            }

            if (opt == "--workspace") {
                if (i + 1 >= argc) {
                    fmt::println("Missing value for --workspace");
                    return false;
                }
                workspace = argv[++i];
                continue;
            }

            if (opt == "--io-type") {
                if (i + 1 >= argc) {
                    fmt::println("Missing value for --io-type");
                    return false;
                }
                std::string v = argv[++i];
                for (auto& c : v) {
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                }
                if (v == "mmap") {
                    ioType = aio::HostBuffer::Strategy::MMAP;
                } else if (v == "alloc") {
                    ioType = aio::HostBuffer::Strategy::ALLOC;
                } else {
                    fmt::println("Unknown io-type: {}. Supported: mmap, alloc", v);
                    return false;
                }
                continue;
            }

            if (opt == "--io-size") {
                if (i + 1 >= argc) {
                    fmt::println("Missing value for --io-size");
                    return false;
                }
                try {
                    ioSize = std::stoull(argv[++i]);
                } catch (const std::exception& e) {
                    fmt::println("Invalid number for --io-size: {}", e.what());
                    return false;
                }
                continue;
            }

            if (opt == "--io-number") {
                if (i + 1 >= argc) {
                    fmt::println("Missing value for --io-number");
                    return false;
                }
                try {
                    ioNumber = std::stoull(argv[++i]);
                } catch (const std::exception& e) {
                    fmt::println("Invalid number for --io-number: {}", e.what());
                    return false;
                }
                continue;
            }

            if (opt == "--device-id") {
                if (i + 1 >= argc) {
                    fmt::println("Missing value for --device-id");
                    return false;
                }
                try {
                    deviceId = std::stoull(argv[++i]);
                } catch (const std::exception& e) {
                    fmt::println("Invalid number for --device-id: {}", e.what());
                    return false;
                }
                continue;
            }

            if (opt == "--epoch-number") {
                if (i + 1 >= argc) {
                    fmt::println("Missing value for --epoch-number");
                    return false;
                }
                try {
                    epochNumber = std::stoull(argv[++i]);
                } catch (const std::exception& e) {
                    fmt::println("Invalid number for --epoch-number: {}", e.what());
                    return false;
                }
                continue;
            }

            fmt::println("Unknown option: {}", opt);
            return false;
        }

        if (workspace.empty()) {
            fmt::println("Error: --workspace is required");
            PrintUsage(argv[0]);
            return false;
        }

        return true;
    }
    void Show() const
    {
        fmt::println("Set Config::Workspace = {}.", workspace);
        fmt::println("Set Config::IoType = {}.",
                     ioType == aio::HostBuffer::Strategy::ALLOC ? "alloc" : "mmap");
        fmt::println("Set Config::IoSize = {}.", ioSize);
        fmt::println("Set Config::IoNumber = {}.", ioNumber);
        fmt::println("Set Config::DeviceId = {}.", deviceId);
        fmt::println("Set Config::EpochNumber = {}.", epochNumber);
    }
};

std::vector<aio::BlockId> MakeBlockIdsRandomly(size_t number)
{
    auto makeBlockIdRandomly = +[] {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<std::uint8_t> dist(0, 255);
        aio::BlockId id;
        for (std::size_t i = 0; i < id.size(); ++i) { id[i] = static_cast<std::byte>(dist(gen)); }
        return id;
    };
    std::vector<aio::BlockId> blockIds;
    blockIds.reserve(number);
    for (size_t i = 0; i < number; i++) { blockIds.push_back(makeBlockIdRandomly()); }
    return blockIds;
}

template <bool write>
void Run(aio::AioEngine& ioEngine, aio::AioEngine::IoTask& task, size_t epochNumber)
{
    using namespace std::chrono;
    const char* taskType = write ? "Write" : "Read";
    const auto size = task.front().length;
    const auto number = task.size();
    const auto total = size * number;
    for (size_t i = 0; i < epochNumber; i++) {
        auto tp = steady_clock::now();
        if constexpr (write) {
            ioEngine.SubmitWrite(task)->Wait();
        } else {
            ioEngine.SubmitRead(task)->Wait();
        }
        auto cost = duration<double>(steady_clock::now() - tp).count() * 1e3;
        auto bandwidth = total / cost / 1e6;
        fmt::println("[{:04}/{:04}] {} task({} x {}) finish, cost={:.3f}ms, bw={:.3f}GB/s.", i + 1,
                     epochNumber, taskType, size, number, cost, bandwidth);
    }
}

int main(int argc, char const* argv[])
{
    Config config;
    if (config.Parse(argc, argv)) {
        config.Show();

        aio::SpaceLayout layout{config.workspace};
        aio::AioEngine ioEngine{&layout, 32};
        aio::HostBuffer buffers{config.ioType, (int32_t)config.deviceId, config.ioSize,
                                config.ioNumber};
        auto blockIds = MakeBlockIdsRandomly(config.ioNumber);
        aio::AioEngine::IoTask task;
        task.reserve(config.ioNumber);
        for (size_t i = 0; i < config.ioNumber; i++) {
            task.push_back(aio::AioEngine::IoShard{blockIds[i], buffers[i], config.ioSize});
        }

        Run<true>(ioEngine, task, config.epochNumber);
        Run<false>(ioEngine, task, config.epochNumber);
        return 0;
    }
    return -1;
}
