/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

/**
 * 设备内存分配器抽象：值缓冲池的内存来源（可移植性约束——设备能力一律接口抽象，
 * 业务侧经本接口分配/写入/读回，不感知具体设备与内存类型）。
 * host 语义为普通进程内存；device 语义为设备内存（显存，经设备 SDK 分配，
 * 读写下须 H2D/D2H 拷贝，地址须对齐到 kv::kAlignmentBytes）。
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
 * deviceSemantics=true 时按 device 语义分配（对齐到 kv::kAlignmentBytes），
 * 用于未编入设备 SDK 环境下以主机内存模拟 device 内存走完整链路。
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
 * 按值缓冲类型创建分配器：host 为主机内存；device 为设备内存
 * （真设备实现经预处理宏 TESTTOOL_ENABLE_ASCEND 挂接，未编入设备 SDK 时以主机内存
 * 模拟 device 语义——memoryType 仍按 DEVICE 注册走完整链路，并打醒目提示日志）
 * @param bufferType conf 的 buffertype 值（host / device），非法值报错可感知
 * @param deviceId conf 的 deviceid 值（buffertype=device 时须非负）
 */
std::unique_ptr<DeviceAllocator> CreateDeviceAllocator(const std::string &bufferType,
                                                       std::int32_t deviceId, std::string &err);
