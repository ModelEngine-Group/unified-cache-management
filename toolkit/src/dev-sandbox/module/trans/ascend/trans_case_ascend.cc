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
#include "trans_case.h"
#include "trans_device_buffer_ascend.h"
#include "trans_host_buffer_ascend.h"
#include "trans_template_ascend.h"

DEFINE_TRANS_H2D_CASE("trans from host(normal) to device(normal) with `ce`", normal, normal, ce,
                      ctx)
{
    TransResult result;
    TransH2DCETemplate transTemplate{ctx.iter, false};
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransHostNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransDeviceNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[H2D] " + brief);
}

DEFINE_TRANS_H2D_CASE("trans from host(anonymous) to device(normal) with `ce`", anonymous, normal,
                      ce, ctx)
{
    TransResult result;
    TransH2DCETemplate transTemplate{ctx.iter, false};
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransHostAnonymousBuffer srcBuffer{device, ctx.size, ctx.num};
        TransDeviceNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[H2D] " + brief);
}

DEFINE_TRANS_H2D_CASE("trans from host(registered) to device(normal) with `ce`", registered, normal,
                      ce, ctx)
{
    TransResult result;
    TransH2DCETemplate transTemplate{ctx.iter, false};
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransHostRegisteredBuffer srcBuffer{device, ctx.size, ctx.num};
        TransDeviceNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[H2D] " + brief);
}

DEFINE_TRANS_D2H_CASE("trans from device(normal) to host(normal) with `ce`", normal, normal, ce,
                      ctx)
{
    TransResult result;
    TransD2HCETemplate transTemplate{ctx.iter, false};
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransDeviceNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransHostNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[D2H] " + brief);
}

DEFINE_TRANS_D2H_CASE("trans from device(normal) to host(anonymous) with `ce`", normal, anonymous,
                      ce, ctx)
{
    TransResult result;
    TransD2HCETemplate transTemplate{ctx.iter, false};
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransDeviceNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransHostAnonymousBuffer dstBuffer{device, ctx.size, ctx.num};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[D2H] " + brief);
}

DEFINE_TRANS_D2H_CASE("trans from device(normal) to host(registered) with `ce`", normal, registered,
                      ce, ctx)
{
    TransResult result;
    TransD2HCETemplate transTemplate{ctx.iter, false};
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransDeviceNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransHostRegisteredBuffer dstBuffer{device, ctx.size, ctx.num};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[D2H] " + brief);
}

DEFINE_TRANS_H2D_CASE("trans from host(normal) to device(normal) with `batch_ce`", normal, normal,
                      batch_ce, ctx)
{
    TransResult result;
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransHostNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransDeviceNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        TransH2DBatchCETemplate transTemplate{device, ctx.iter, false};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[H2D] " + brief);
}

DEFINE_TRANS_H2D_CASE("trans from host(registered) to device(normal) with `batch_ce`", registered,
                      normal, batch_ce, ctx)
{
    TransResult result;
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransHostRegisteredBuffer srcBuffer{device, ctx.size, ctx.num};
        TransDeviceNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        TransH2DBatchCETemplate transTemplate{device, ctx.iter, false};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[H2D] " + brief);
}

DEFINE_TRANS_D2H_CASE("trans from device(normal) to host(normal) with `batch_ce`", normal, normal,
                      batch_ce, ctx)
{
    TransResult result;
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransDeviceNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransHostNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        TransD2HBatchCETemplate transTemplate{device, ctx.iter, false};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[D2H] " + brief);
}

DEFINE_TRANS_D2H_CASE("trans from device(normal) to host(registered) with `batch_ce`", normal,
                      registered, batch_ce, ctx)
{
    TransResult result;
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransDeviceNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransHostRegisteredBuffer dstBuffer{device, ctx.size, ctx.num};
        TransD2HBatchCETemplate transTemplate{device, ctx.iter, false};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[D2H] " + brief);
}

DEFINE_TRANS_H2D_CASE("trans from host(normal) to device(normal) with `ms_48`", normal, normal,
                      ms_48, ctx)
{
    constexpr std::size_t streamCount = 48;
    TransResult result;
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransHostNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransDeviceNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        TransH2DMultiStreamTemplate transTemplate{streamCount, ctx.iter, false};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[H2D] " + brief);
}

DEFINE_TRANS_H2D_CASE("trans from host(registered) to device(normal) with `ms_48`", registered,
                      normal, ms_48, ctx)
{
    constexpr std::size_t streamCount = 48;
    TransResult result;
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransHostRegisteredBuffer srcBuffer{device, ctx.size, ctx.num};
        TransDeviceNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        TransH2DMultiStreamTemplate transTemplate{streamCount, ctx.iter, false};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[H2D] " + brief);
}

DEFINE_TRANS_D2H_CASE("trans from device(normal) to host(normal) with `ms_48`", normal, normal,
                      ms_48, ctx)
{
    constexpr std::size_t streamCount = 48;
    TransResult result;
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransDeviceNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransHostNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        TransD2HMultiStreamTemplate transTemplate{streamCount, ctx.iter, false};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[D2H] " + brief);
}

DEFINE_TRANS_D2H_CASE("trans from device(normal) to host(registered) with `ms_48`", normal,
                      registered, ms_48, ctx)
{
    constexpr std::size_t streamCount = 48;
    TransResult result;
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransDeviceNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransHostRegisteredBuffer dstBuffer{device, ctx.size, ctx.num};
        TransD2HMultiStreamTemplate transTemplate{streamCount, ctx.iter, false};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[D2H] " + brief);
}
