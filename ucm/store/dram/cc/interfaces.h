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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_INTERFACES_H
#define UNIFIEDCACHE_DRAM_STORE_CC_INTERFACES_H

#include "messages.h"
#include "status/status.h"

namespace UC::Dram {

// Command interface
class INodeCommandSink {
public:
    virtual ~INodeCommandSink() = default;
    virtual Status Post(NodeId nodeId, NodeCommand&& command) = 0;
};

class ITransportCommandSink {
public:
    virtual ~ITransportCommandSink() = default;
    virtual Status Post(NodeId nodeId, TransportCommand&& command) = 0;
};

class IReplyWatcher {
public:
    virtual ~IReplyWatcher() = default;
    virtual Status Watch(WatchReply watch) = 0;
    virtual void Unwatch(const ReplySlot& slot) = 0;
};

// Event interface
class ITaskEventSink {
public:
    virtual ~ITaskEventSink() = default;
    virtual bool TryPost(TaskEvent&& event) = 0;
};

class INodeEventSink {
public:
    virtual ~INodeEventSink() = default;
    virtual bool TryPost(NodeId nodeId, NodeEvent&& event) = 0;
};

}  // namespace UC::Dram

#endif
