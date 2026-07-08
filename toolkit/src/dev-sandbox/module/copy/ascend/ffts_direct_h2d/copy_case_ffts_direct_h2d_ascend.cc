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
#include <acl/acl.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include "ascend/copy_buffer_ascend.h"
#include "ascend/forked_copy_runner_ascend.h"
#include "copy_case.h"
#include "copy_instance_ffts_direct_h2d_ascend.h"
#include "mapped_host_buffer_ffts_direct_h2d_ascend.h"

namespace {

bool FftsDirectValidationEnabled()
{
    const char* value = std::getenv("COPY_FFTS_VALIDATE");
    if (value == nullptr) { return false; }
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "on") == 0 ||
           std::strcmp(value, "ON") == 0;
}

std::vector<uint8_t> MakeFftsDirectPattern(size_t fragmentIndex, size_t size)
{
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>((fragmentIndex * 17 + i * 31 + (i >> 8)) & 0xFF);
    }
    return data;
}

void InitializeFftsDirectHostPatternedBuffer(const CopyBuffer& buffer)
{
    for (size_t i = 0; i < buffer.Number(); ++i) {
        const auto pattern = MakeFftsDirectPattern(i, buffer.Size());
        std::memcpy(buffer[i], pattern.data(), pattern.size());
    }
}

void ResetFftsDirectDeviceBuffer(const CopyBuffer& buffer)
{
    ASCEND_ASSERT(aclrtSetDevice(buffer.Device()));
    for (size_t i = 0; i < buffer.Number(); ++i) {
        ASCEND_ASSERT(aclrtMemset(buffer[i], buffer.Size(), 0, buffer.Size()));
    }
}

std::vector<uint8_t> CopyFftsDirectDeviceToHost(const CopyBuffer& buffer, size_t index)
{
    std::vector<uint8_t> data(buffer.Size());
    ASCEND_ASSERT(aclrtSetDevice(buffer.Device()));
    ASCEND_ASSERT(aclrtMemcpy(data.data(), data.size(), buffer[index], buffer.Size(),
                              ACL_MEMCPY_DEVICE_TO_HOST));
    return data;
}

bool ValidateFftsDirectPatternedBuffer(const CopyBuffer& buffer)
{
    for (size_t i = 0; i < buffer.Number(); ++i) {
        auto actual = CopyFftsDirectDeviceToHost(buffer, i);
        auto expected = MakeFftsDirectPattern(i, buffer.Size());
        if (actual.size() != expected.size() ||
            std::memcmp(actual.data(), expected.data(), expected.size()) != 0) {
            return false;
        }
    }
    return true;
}

void ValidateFftsDirectDeviceBufferIfEnabled(const CopyBuffer& buffer, bool enabled)
{
    if (enabled) { ASSERT(ValidateFftsDirectPatternedBuffer(buffer)); }
}

void PrintFftsDirectValidationPassIfEnabled(const CopyCase& copyCase, bool enabled)
{
    if (enabled) { std::cout << "[validation] " << copyCase.Key() << " PASS\n"; }
}

size_t FftsDirectTotalFragments(const CopyCase::Context& ctx)
{
    ASSERT(ctx.num > 0);
    if (ctx.frags == 0) { return ctx.num; }
    ASSERT(ctx.num <= std::numeric_limits<size_t>::max() / ctx.frags);
    return ctx.num * ctx.frags;
}

}  // namespace

DEFINE_COPY_CASE_NO_RUNTIME(
    AllHost2AllDeviceFftsDirectH2DCase, "all_host_to_all_device_ffts_direct_h2d",
    "copy all aclrtMallocHost mapped host buffers to all device buffers with ffts direct h2d",
    ctx)
{
    CopyResult result;
    const bool validationEnabled = FftsDirectValidationEnabled();
    result.Push(ascend_copy::RunForkedCopyBatch(
        ctx, "acl::host_mapped::all", "acl::device::all", "ffts-direct-h2d",
        [&](size_t device) {
            const auto fragments = FftsDirectTotalFragments(ctx);
            FftsMappedHostCopyBuffer srcBuffer{device, ctx.size, fragments};
            DeviceCopyBuffer dstBuffer{device, ctx.size, fragments};
            InitializeFftsDirectHostPatternedBuffer(srcBuffer);
            ResetFftsDirectDeviceBuffer(dstBuffer);

            FftsDirectH2DCopyInstance instance{ctx.iter, false, ctx.frags};
            auto childResult = instance.DoCopy(&srcBuffer, &dstBuffer);
            ValidateFftsDirectDeviceBufferIfEnabled(dstBuffer, validationEnabled);
            return childResult;
        }));
    PrintFftsDirectValidationPassIfEnabled(*this, validationEnabled);
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE_NO_RUNTIME(
    AllODirectHost2AllDeviceFftsDirectH2DCase,
    "all_odirect_host_to_all_device_ffts_direct_h2d",
    "copy all UCM O_DIRECT style mmap mapped host buffers to all device buffers with "
    "ffts direct h2d",
    ctx)
{
    CopyResult result;
    const bool validationEnabled = FftsDirectValidationEnabled();
    result.Push(ascend_copy::RunForkedCopyBatch(
        ctx, "acl::odirect_mmap::all", "acl::device::all", "ffts-direct-h2d",
        [&](size_t device) {
            const auto fragments = FftsDirectTotalFragments(ctx);
            FftsODirectMappedHostCopyBuffer srcBuffer{device, ctx.size, fragments};
            DeviceCopyBuffer dstBuffer{device, ctx.size, fragments};
            InitializeFftsDirectHostPatternedBuffer(srcBuffer);
            ResetFftsDirectDeviceBuffer(dstBuffer);

            FftsDirectH2DCopyInstance instance{ctx.iter, false, ctx.frags};
            auto childResult = instance.DoCopy(&srcBuffer, &dstBuffer);
            ValidateFftsDirectDeviceBufferIfEnabled(dstBuffer, validationEnabled);
            return childResult;
        }));
    PrintFftsDirectValidationPassIfEnabled(*this, validationEnabled);
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE_NO_RUNTIME(
    OneShareHost2AllDeviceFftsDirectH2DCase, "one_share_host_to_all_device_ffts_direct_h2d",
    "copy one shared mapped host buffer to all device buffers with ffts direct h2d using fork submit",
    ctx)
{
    CopyResult result;
    const bool validationEnabled = FftsDirectValidationEnabled();
    const auto fragments = FftsDirectTotalFragments(ctx);
    FftsMappedSharedHostRegion srcRegion{"one_share_host_to_all_device_ffts_direct_h2d", 0,
                                         ctx.size, fragments};
    InitializeFftsDirectHostPatternedBuffer(srcRegion);
    result.Push(ascend_copy::RunForkedCopyBatch(
        ctx, srcRegion.Name(), "acl::device::all", "ffts-direct-h2d", [&](size_t device) {
            FftsMappedSharedHostCopyBuffer srcBuffer{srcRegion.ShmName(),
                                                    srcRegion.MappedBytes(), device, ctx.size,
                                                    fragments};
            DeviceCopyBuffer dstBuffer{device, ctx.size, fragments};
            ResetFftsDirectDeviceBuffer(dstBuffer);

            FftsDirectH2DCopyInstance instance{ctx.iter, false, ctx.frags};
            auto childResult = instance.DoCopy(&srcBuffer, &dstBuffer);
            ValidateFftsDirectDeviceBufferIfEnabled(dstBuffer, validationEnabled);
            return childResult;
        }));
    PrintFftsDirectValidationPassIfEnabled(*this, validationEnabled);
    result.Show("[[ " + Key() + " ]] " + Brief());
}
