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
#include "copy_buffer_ascend.h"
#include "copy_case.h"
#include "copy_instance_ascend.h"
#include "forked_copy_runner_ascend.h"

DEFINE_COPY_CASE(Host2DeviceCECase, "host_to_device_ce",
                 "memcpy from host to device with ce one by one", ctx)
{
    CopyResult result;
    for (size_t device = 0; device < ctx.nDevice; device++) {
        HostCopyBuffer srcBuffer{device, ctx.size, ctx.num};
        DeviceCopyBuffer dstBuffer{device, ctx.size, ctx.num};
        H2DCECopyInstance instance{ctx.iter, false};
        result.Push(instance.DoCopy(&srcBuffer, &dstBuffer));
    }
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE(Host2DeviceBatchCECase, "host_to_device_batch_ce",
                 "memcpy from host to device with batch ce one by one", ctx)
{
    CopyResult result;
    for (size_t device = 0; device < ctx.nDevice; device++) {
        HostCopyBuffer srcBuffer{device, ctx.size, ctx.num};
        DeviceCopyBuffer dstBuffer{device, ctx.size, ctx.num};
        H2DBatchCECopyInstance instance{ctx.iter, false, device};
        result.Push(instance.DoCopy(&srcBuffer, &dstBuffer));
    }
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE(OneHost2AllDeviceCECase, "one_host_to_all_device_ce",
                 "memcpy from one host to all device with ce", ctx)
{
    CopyResult result;
    HostCopyBuffer srcBuffer{0, ctx.size, ctx.num};
    for (size_t device = 0; device < ctx.nDevice; device++) {
        DeviceCopyBuffer dstBuffer{device, ctx.size, ctx.num};
        H2DCECopyInstance instance{ctx.iter, false};
        result.Push(instance.DoCopy(&srcBuffer, &dstBuffer));
    }
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE(AllHost2AllDeviceCECase, "all_host_to_all_device_ce",
                 "memcpy from all host to all device with ce at one time", ctx)
{
    // Use unique_ptr for automatic memory management (RAII)
    std::vector<std::unique_ptr<CopyBuffer>> srcBuffers(ctx.nDevice);
    std::vector<std::unique_ptr<CopyBuffer>> dstBuffers(ctx.nDevice);

    for (size_t device = 0; device < ctx.nDevice; device++) {
        srcBuffers[device] = std::make_unique<HostCopyBuffer>(device, ctx.size, ctx.num);
        dstBuffers[device] = std::make_unique<DeviceCopyBuffer>(device, ctx.size, ctx.num);
    }

    // Extract raw pointers for DoCopyBatch (which requires raw pointer vectors)
    std::vector<const CopyBuffer*> srcPtrs;
    std::vector<const CopyBuffer*> dstPtrs;
    srcPtrs.reserve(ctx.nDevice);
    dstPtrs.reserve(ctx.nDevice);
    for (size_t device = 0; device < ctx.nDevice; device++) {
        srcPtrs.push_back(srcBuffers[device].get());
        dstPtrs.push_back(dstBuffers[device].get());
    }

    H2DCECopyInstance instance{ctx.iter, false};
    CopyResult result;
    result.Push(instance.DoCopyBatch(srcPtrs, dstPtrs));

    // No manual delete needed - unique_ptr automatically releases memory on destruction
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE(Device2DeviceCECase, "device_to_device_ce",
                 "memcpy from device to device with ce one by one", ctx)
{
    CopyResult result;
    for (size_t device = 0; device < ctx.nDevice; device++) {
        DeviceCopyBuffer srcBuffer{device, ctx.size, ctx.num};
        DeviceCopyBuffer dstBuffer{device, ctx.size, ctx.num};
        D2DCECopyInstance instance{ctx.iter, false};
        result.Push(instance.DoCopy(&srcBuffer, &dstBuffer));
    }
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE(OneDevice2AllDeviceCECase, "one_device_to_all_device_ce",
                 "memcpy from one device to all device with ce", ctx)
{
    CopyResult result;
    DeviceCopyBuffer srcBuffer{0, ctx.size, ctx.num};
    for (size_t device = 0; device < ctx.nDevice; device++) {
        DeviceCopyBuffer dstBuffer{device, ctx.size, ctx.num};
        D2DCECopyInstance instance{ctx.iter, false};
        result.Push(instance.DoCopy(&srcBuffer, &dstBuffer));
    }
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE(Anonymous2DeviceCECase, "anonymous_to_device_ce",
                 "memcpy from anonymous to device one by one", ctx)
{
    CopyResult result;
    for (size_t device = 0; device < ctx.nDevice; device++) {
        AnonymousCopyBuffer srcBuffer{device, ctx.size, ctx.num};
        DeviceCopyBuffer dstBuffer{device, ctx.size, ctx.num};
        H2DCECopyInstance instance{ctx.iter, false};
        result.Push(instance.DoCopy(&srcBuffer, &dstBuffer));
    }
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE(Host2DeviceCEMultiStreamCase, "host_to_device_ce_multi_stream",
                 "memcpy from host to device with ce using multi stream one by one", ctx)
{
    constexpr auto streamCount = 4;
    CopyResult result;
    for (size_t device = 0; device < ctx.nDevice; device++) {
        HostCopyBuffer srcBuffer{device, ctx.size, ctx.num};
        DeviceCopyBuffer dstBuffer{device, ctx.size, ctx.num};
        H2DCEMultiStreamCopyInstance instance{ctx.iter, false, streamCount};
        result.Push(instance.DoCopy(&srcBuffer, &dstBuffer));
    }
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE_NO_RUNTIME(
    OneShareHost2AllDeviceCEMultiStreamCase, "one_share_host_to_all_device_ce_multi_stream",
    "memcpy from one shared host to all device with ce using multi stream and fork submit", ctx)
{
    constexpr auto streamCount = 4;
    CopyResult result;
    SharedHostRegion srcRegion{"one_share_host_to_all_device_ce_multi_stream", 0, ctx.size,
                               ctx.num};
    result.Push(ascend_copy::RunForkedCopyBatch(
        ctx, srcRegion.Name(), "acl::device::all", "CE-MS-FORK", [&](size_t device) {
            SharedHostCopyBuffer srcBuffer{srcRegion.ShmName(), device, ctx.size, ctx.num};
            DeviceCopyBuffer dstBuffer{device, ctx.size, ctx.num};
            H2DCEMultiStreamCopyInstance instance{ctx.iter, false, streamCount};
            return instance.DoCopy(&srcBuffer, &dstBuffer);
        }));
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE_NO_RUNTIME(
    AllHost2AllDeviceCEMultiStreamCase, "all_host_to_all_device_ce_multi_stream",
    "memcpy from all host to all device with ce using multi stream and fork submit", ctx)
{
    constexpr auto streamCount = 4;
    CopyResult result;
    result.Push(ascend_copy::RunForkedCopyBatch(
        ctx, "acl::host::all", "acl::device::all", "CE-MS-FORK", [&](size_t device) {
            HostCopyBuffer srcBuffer{device, ctx.size, ctx.num};
            DeviceCopyBuffer dstBuffer{device, ctx.size, ctx.num};
            H2DCEMultiStreamCopyInstance instance{ctx.iter, false, streamCount};
            return instance.DoCopy(&srcBuffer, &dstBuffer);
        }));
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE_NO_RUNTIME(
    AllODirectHost2AllDeviceCEMultiStreamCase, "all_odirect_host_to_all_device_ce_multi_stream",
    "memcpy from all UCM O_DIRECT style host to all device with ce using multi stream and fork "
    "submit",
    ctx)
{
    constexpr auto streamCount = 4;
    CopyResult result;
    result.Push(ascend_copy::RunForkedCopyBatch(
        ctx, "acl::odirect_mmap::all", "acl::device::all", "CE-MS-FORK", [&](size_t device) {
            ODirectHostCopyBuffer srcBuffer{device, ctx.size, ctx.num};
            DeviceCopyBuffer dstBuffer{device, ctx.size, ctx.num};
            H2DCEMultiStreamCopyInstance instance{ctx.iter, false, streamCount};
            return instance.DoCopy(&srcBuffer, &dstBuffer);
        }));
    result.Show("[[ " + Key() + " ]] " + Brief());
}
