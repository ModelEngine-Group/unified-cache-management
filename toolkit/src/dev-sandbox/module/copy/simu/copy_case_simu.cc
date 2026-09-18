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
#include "copy_buffer_simu.h"
#include "copy_case.h"
#include "copy_instance_simu.h"

DEFINE_COPY_CASE(Host2AnonymousMemcpyCase, "host_to_anonymous_memcpy",
                 "memcpy from host to anonymous one by one", ctx)
{
    CopyResult result;
    for (size_t device = 0; device < ctx.nDevice; device++) {
        HostCopyBuffer srcBuffer{device, ctx.size, ctx.num};
        AnonymousCopyBuffer dstBuffer{device, ctx.size, ctx.num};
        MemcpyCopyInstance instance{ctx.iter, false};
        result.Push(instance.DoCopy(&srcBuffer, &dstBuffer));
    }
    result.Show("[[ " + Key() + " ]] " + Brief());
}

DEFINE_COPY_CASE(Shm2AllHostMemcpyCase, "shm_to_all_host_memcpy",
                 "memcpy from shm to all host at one time", ctx)
{
    MemcpyCopyInstance instance{ctx.iter, true};
    CopyResult result;
    std::string shmKey = Key() + ".shm";
    const char* shmFile = shmKey.c_str();
    std::vector<const CopyBuffer*> srcBuffers(ctx.nDevice, nullptr);
    std::vector<const CopyBuffer*> dstBuffers(ctx.nDevice, nullptr);
    for (size_t device = 0; device < ctx.nDevice; device++) {
        srcBuffers[device] = new ShmCopyBuffer(shmFile, device, ctx.size, ctx.num);
        dstBuffers[device] = new HostCopyBuffer(device, ctx.size, ctx.num);
    }
    result.Push(instance.DoCopyBatch(srcBuffers, dstBuffers));
    result.Show("[[ " + Key() + " ]] " + Brief());
    for (size_t device = 0; device < ctx.nDevice; device++) {
        delete srcBuffers[device];
        delete dstBuffers[device];
    }
}
