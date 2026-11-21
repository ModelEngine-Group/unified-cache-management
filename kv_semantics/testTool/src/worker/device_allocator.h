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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

/**
 * 设备内存分配器抽象：值缓冲池的内存来源（可移植性约束——设备能力一律接口抽象，
 * 业务侧经本接口分配/写入/读回，不感知具体设备与内存类型）。
 * cpu 语义为普通进程内存（注册 MemoryType::HOST）；gpu 语义为 CUDA 显存、
 * npu 语义为 ACL NPU 内存（均注册 MemoryType::DEVICE，读写下须 H2D/D2H 拷贝，
 * 地址须对齐到 kv::kAlignmentBytes=512）。
 */
class DeviceAllocator {
public:
    virtual ~DeviceAllocator() = default;

    /**
     * 分配 size 字节内存块
     * @param addr 带出内存块地址（device 语义下对齐到 kv::kAlignmentBytes）
     * @param err 失败时的错误描述
     */
    virtual bool Allocate(std::size_t size, std::uint64_t &addr, std::string &err) = 0;

    /** 释放 Allocate 分配的内存块；addr 为 0 时安全跳过 */
    virtual void Deallocate(std::uint64_t addr) = 0;

    /** 把 host 数据写入内存块（device 语义下为 H2D 拷贝） */
    virtual bool Write(std::uint64_t addr, const void *hostData, std::size_t size,
                       std::string &err) = 0;

    /** 把内存块数据读回 host 内存（device 语义下为 D2H 拷贝） */
    virtual bool Read(void *hostData, std::uint64_t addr, std::size_t size, std::string &err) = 0;

    /** 是否 device 语义内存（决定注册 MemoryRegion 的 memoryType 与提示信息） */
    virtual bool IsDeviceMemory() const = 0;
};

/**
 * 主机内存分配器：malloc/free + memcpy。
 * deviceSemantics=true 时按 device 语义分配（对齐到 kv::kAlignmentBytes，
 * IsDeviceMemory 返回 true，注册按 MemoryType::DEVICE），用于：
 * ① buffertype=cpu 的常规主机内存（deviceSemantics=false）；
 * ② 未编入设备 SDK（TESTTOOL_ENABLE_CUDA / TESTTOOL_ENABLE_ASCEND）时
 *    gpu/npu 以主机内存模拟 device 语义走完整链路。
 */
class HostAllocator : public DeviceAllocator {
public:
    /** @param deviceSemantics true 时按 device 语义分配（地址对齐、注册类型提示） */
    explicit HostAllocator(bool deviceSemantics);

    bool Allocate(std::size_t size, std::uint64_t &addr, std::string &err) override;
    void Deallocate(std::uint64_t addr) override;
    bool Write(std::uint64_t addr, const void *hostData, std::size_t size,
               std::string &err) override;
    bool Read(void *hostData, std::uint64_t addr, std::size_t size, std::string &err) override;
    bool IsDeviceMemory() const override;

private:
    /** 是否按 device 语义分配（地址对齐与注册类型提示） */
    const bool deviceSemantics_;
};

/**
 * 按值缓冲类型创建分配器（GPU/NPU 真实现类在 .cc 内宏隔离，头文件零 SDK 依赖）：
 * cpu 为主机内存（MemoryType::HOST）；gpu 为 CUDA 显存、npu 为 ACL NPU 内存
 * （均 MemoryType::DEVICE；真实现分别经预处理宏 TESTTOOL_ENABLE_CUDA /
 * TESTTOOL_ENABLE_ASCEND 编入，未编入时以主机内存模拟 device 语义——memoryType
 * 仍按 DEVICE 注册走完整链路，并打醒目提示日志）
 * @param bufferType conf 的 buffertype 值（cpu / gpu / npu），非法值报错可感知
 * @param deviceId conf 的 deviceid 值（gpu/npu 时须非负，作为 CUDA device / ACL device 编号）
 */
std::unique_ptr<DeviceAllocator> CreateDeviceAllocator(const std::string &bufferType,
                                                       std::int32_t deviceId, std::string &err);
