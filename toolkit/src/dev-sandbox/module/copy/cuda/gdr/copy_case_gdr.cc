/**
 * MIT License
 *
 * Copyright (c) 2026 relat-ivity
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
#include "copy_buffer_gdr.h"
#include "copy_case.h"
#include "copy_instance_gdr.h"

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "error_handle_gdr.h"
#include "rdma_channel_gdr.h"
#include "submit_executor_gdr.h"

namespace {
std::vector<std::string> SplitNicNames(const char* nicNames)
{
    std::vector<std::string> result;
    if (nicNames == nullptr) { return result; }

    std::stringstream stream{nicNames};
    std::string nicName;
    while (std::getline(stream, nicName, ',')) {
        if (!nicName.empty()) { result.push_back(nicName); }
    }
    return result;
}

std::vector<std::string> DefaultNicNames()
{
    return {"mlx5_0", "mlx5_2", "mlx5_4", "mlx5_6", "mlx5_8", "mlx5_10", "mlx5_12", "mlx5_14"};
}

class GdrRuntimeGuard {
public:
    explicit GdrRuntimeGuard(size_t deviceNumber)
    {
        RDMAChannelConfig config;
        config.cqDepth = 1024;
        config.qpSendWr = 1024;
        config.qpRecvWr = 1024;

        auto nicNames = SplitNicNames(std::getenv("GDR_NICS"));
        if (nicNames.empty()) { nicNames = DefaultNicNames(); }
        ASSERT(nicNames.size() == deviceNumber);

        SubmitExecutor::Instance().Initialize(deviceNumber);
        ChannelManager::Instance().Initialize(static_cast<int32_t>(deviceNumber), nicNames, config);
    }

    ~GdrRuntimeGuard()
    {
        SubmitExecutor::Instance().Shutdown();
        ChannelManager::Instance().Shutdown();
    }

    GdrRuntimeGuard(const GdrRuntimeGuard&) = delete;
    GdrRuntimeGuard& operator=(const GdrRuntimeGuard&) = delete;
};
}  // namespace

DEFINE_COPY_CASE(Host2DeviceGdrCase, "host_to_device_gdr",
                 "memcpy from host to device with gdr one by one", ctx)
{
    GdrRuntimeGuard guard{ctx.nDevice};
    CopyResult result;
    for (size_t device = 0; device < ctx.nDevice; device++) {
        GdrHostCopyBuffer srcBuffer{device, ctx.size, ctx.num};
        GdrDeviceCopyBuffer dstBuffer{device, ctx.size, ctx.num};
        GdrCopyInstance instance{ctx.iter};
        result.Push(instance.DoCopy(&srcBuffer, &dstBuffer));
    }
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE(OneHost2AllDeviceGdrCase, "one_host_to_all_device_gdr",
                 "memcpy from one host to all device with gdr", ctx)
{
    GdrRuntimeGuard guard{ctx.nDevice};
    GdrHostCopyBuffer srcBuffer{0, ctx.size, ctx.num};
    std::vector<const CopyBuffer*> srcBuffers(ctx.nDevice, &srcBuffer);
    std::vector<const CopyBuffer*> dstBuffers(ctx.nDevice);
    for (size_t device = 0; device < ctx.nDevice; device++) {
        dstBuffers[device] = new GdrDeviceCopyBuffer{device, ctx.size, ctx.num};
    }
    GdrCopyInstance instance{ctx.iter};
    CopyResult result;
    result.Push(instance.DoCopyBatch(srcBuffers, dstBuffers));
    for (size_t device = 0; device < ctx.nDevice; device++) { delete dstBuffers[device]; }
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE(AllHost2AllDeviceGdrCase, "all_host_to_all_device_gdr",
                 "memcpy from all host to all device with gdr at one time", ctx)
{
    GdrRuntimeGuard guard{ctx.nDevice};
    std::vector<const CopyBuffer*> srcBuffers(ctx.nDevice);
    std::vector<const CopyBuffer*> dstBuffers(ctx.nDevice);
    for (size_t device = 0; device < ctx.nDevice; device++) {
        srcBuffers[device] = new GdrHostCopyBuffer{device, ctx.size, ctx.num};
        dstBuffers[device] = new GdrDeviceCopyBuffer{device, ctx.size, ctx.num};
    }
    GdrCopyInstance instance{ctx.iter};
    CopyResult result;
    result.Push(instance.DoCopyBatch(srcBuffers, dstBuffers));
    for (size_t device = 0; device < ctx.nDevice; device++) {
        delete srcBuffers[device];
        delete dstBuffers[device];
    }
    result.Show("[[ " + Key() + " ]] " + Brief());
}
