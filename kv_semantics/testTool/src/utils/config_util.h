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

#include <cstdint>
#include <spdlog/common.h>
#include <string>
#include <unordered_map>

/**
 * 全局配置中心（单例）：加载配置文件，原始键值以 string 存入缓存，getter 按需解析为目标类型；
 * 数值项按实际值域选型：计数/模式/ID 等小值域用 int，字节量（buffersize、log.max_file_size）用
 * int64； getter 采用安全访问 + 抛异常：key 缺失或数值解析失败抛 std::runtime_error（信息含 key 与
 * 原始值），由 main 统一 catch 报错退出；Load/Check 保持 bool + err 契约（内部自行兜异常），
 * Check 校验级别名合法、runtime=0 时 loop>0 与值缓冲池参数为正。
 * 需要读配置的组件在初始化入口显式传入实例引用，不在内部直接访问单例。
 */
class ConfigUtil {
public:
    /** 获取全局单例引用（C++11 局部静态变量，首次调用时初始化且线程安全） */
    static ConfigUtil &GetInstance();

    /** 加载配置（解析 key=value、剥离行内注释、Check
     * 校验、展开文件名时间标记）；任一环节失败则整体失败 */
    bool Load(const std::string &path, std::string &err);

    // ---- 测试通用参数 ----
    /** 输入文件类型（testmode）：1=不带 value，2=带 value */
    int GetTestMode() const;
    /** 运行时长秒（runtime）；0 为循环模式，以 loop 为准 */
    int GetRuntime() const;
    /** 请求循环执行次数（loop），runtime=0 时生效 */
    int GetLoop() const;
    /** 性能模式（perf）：true=压测模式（主 logger 级别仍取 log.level 配置，请求明细日志关闭） */
    bool GetPerf() const;
    /** 只建链不下 IO（onlyconnect） */
    bool GetOnlyConnect() const;
    /** 建链后自动断开时长秒（autodisconnect，可选键）：0（缺省）禁用；>0 建链成功 n
     * 秒后自动断开并退出，仅 onlyconnect=on 时生效 */
    int GetAutoDisconnect() const;
    /** 绑核起始 ID（startcoreid） */
    int GetStartCoreId() const;
    /** 发送线程数量（numjobs） */
    int GetNumJobs() const;
    /** 值缓冲单块容量字节（buffersize），建链时分配注册一次；字节量可能超 int 值域，留 int64 */
    int64_t GetBufferSize() const;
    /** 值缓冲块数（buffercount），建链时分配注册一次 */
    int GetBufferCount() const;
    /** 值缓冲内存类型（buffertype）：host 主机内存 / device 设备内存 */
    std::string GetBufferType() const;
    /** 设备 ID（deviceid），buffertype=device 时使用 */
    int GetDeviceId() const;
    /** 读 log 的位置（readlogdir） */
    std::string GetReadLogDir() const;
    /** 日志输出目录（logdir） */
    std::string GetLogDir() const;

    // ---- log 模块参数 ----
    /** 全量日志最低级别（log.level） */
    spdlog::level::level_enum GetLogLevel() const;
    /** 全量日志是否同步打印到控制台（log.console，可选键）：缺省或 on 时打开，off 关闭 */
    bool GetLogConsole() const;
    /** 日志输出格式（log.pattern），语法同 spdlog pattern，值不能含 '#' */
    std::string GetLogPattern() const;
    /** 全量日志实际文件名（log.file_name 的 %Y%m%d_%H 等时间标记已按加载时刻展开） */
    std::string GetLogFileName() const;
    /** warn 专用实际文件名（log.warn_file_name，时间标记已展开），记录 warn 及以上 */
    std::string GetLogWarnFileName() const;
    /** error 专用实际文件名（log.error_file_name，时间标记已展开） */
    std::string GetLogErrorFileName() const;
    /** 单文件大小上限字节（log.max_file_size）；字节量可能超 int 值域，留 int64 */
    int64_t GetLogMaxFileSize() const;
    /** 滚动文件保留个数，不含活跃文件（log.max_files），0 表示不清理 */
    int GetLogMaxFiles() const;
    /** 异步队列长度（log.async_queue_size） */
    int GetLogAsyncQueueSize() const;
    /** 周期刷盘间隔毫秒（log.async_flush_interval_ms） */
    int GetLogAsyncFlushIntervalMs() const;

    /** 将全部配置项经 getter 读出并以 info 级别输出到日志；须在 Load 与日志初始化成功后调用 */
    void Print() const;

    ConfigUtil(const ConfigUtil &) = delete;             // 单例：禁止拷贝
    ConfigUtil &operator=(const ConfigUtil &) = delete;  // 单例：禁止赋值

private:
    ConfigUtil() = default;  // 单例：私有默认构造，仅 GetInstance 内部创建

    /** 取 key 对应原始字符串值；key 缺失时抛 std::runtime_error（信息含 key 名） */
    std::string RequireValue(const std::string &key) const;

    /** 取 key 对应 int 值（stoi 解析）；key 缺失或解析失败抛 std::runtime_error（信息含 key
     * 与原始值） */
    int RequireInt(const std::string &key) const;

    /** 取 key 对应 int64 值（stoll 解析）；key 缺失或解析失败抛 std::runtime_error（信息含 key
     * 与原始值） */
    std::int64_t RequireInt64(const std::string &key) const;

    /** 配置检查：级别名合法、runtime=0 时 loop 须大于
     * 0、值缓冲池参数（buffersize/buffercount）须为正；发现问题填 err 并返回 false */
    bool Check(std::string &err);

    /** 原始键值缓存（全 string；文件名值为加载时刻展开后的实际文件名） */
    std::unordered_map<std::string, std::string> values_;
};
