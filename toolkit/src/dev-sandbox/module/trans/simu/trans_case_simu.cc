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
#include "trans_buffer_simu.h"
#include "trans_case.h"
#include "trans_template_simu.h"

DEFINE_TRANS_H2D_CASE("trans form host(normal) to device(normal) with `memcpy`", normal, normal,
                      memcpy, ctx)
{
    TransMemcpyTemplate transTemplate{ctx.iter, false};
    TransResult result;
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransHostNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransDeviceNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[H2D] " + brief);
}

DEFINE_TRANS_H2D_CASE("trans form host(anonymous) to device(normal) with `memcpy`", anonymous,
                      normal, memcpy, ctx)
{
    TransMemcpyTemplate transTemplate{ctx.iter, false};
    TransResult result;
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransHostAnonymousBuffer srcBuffer{device, ctx.size, ctx.num};
        TransDeviceNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[H2D] " + brief);
}

DEFINE_TRANS_D2H_CASE("trans form device(normal) to host(normal) with `memcpy`", normal, normal,
                      memcpy, ctx)
{
    TransMemcpyTemplate transTemplate{ctx.iter, false};
    TransResult result;
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransDeviceNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransHostNormalBuffer dstBuffer{device, ctx.size, ctx.num};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[D2H] " + brief);
}

DEFINE_TRANS_D2H_CASE("trans form device(normal) to host(anonymous) with `memcpy`", normal,
                      anonymous, memcpy, ctx)
{
    TransMemcpyTemplate transTemplate{ctx.iter, false};
    TransResult result;
    for (std::size_t device = 0; device < ctx.nDevice; device++) {
        TransDeviceNormalBuffer srcBuffer{device, ctx.size, ctx.num};
        TransHostAnonymousBuffer dstBuffer{device, ctx.size, ctx.num};
        result.Push(transTemplate.TransOne(&srcBuffer, &dstBuffer));
    }
    result.Show("[D2H] " + brief);
}
