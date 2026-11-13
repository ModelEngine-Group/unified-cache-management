/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include <spdlog/common.h>

/**
 * 全局配置中心：加载 conf/test_config.conf，原始键值以 string 存入缓存，
 * key 齐全性与取值边界在 Load 时经 Check 校验，getter 按需解析为目标类型。
 */
class ConfigUtil {
public:
    /** 加载并校验全部配置（解析 key=value、剥离行内注释、Check 校验、展开文件名时间标记）；任一环节失败则整体失败 */
    static bool Load(const std::string &path, std::string &err);

    // ---- 测试通用参数 ----
    /** 输入文件类型（testmode）：1=不带 value，2=带 value */
    static int64_t GetTestMode();
    /** 运行时长分钟（runtime）；0 为循环模式，以 loop 为准 */
    static int64_t GetRuntime();
    /** 请求循环执行次数（loop），runtime=0 时生效 */
    static int64_t GetLoop();
    /** 性能模式（perf）：true=无日志测性能 */
    static bool GetPerf();
    /** 只建链不下 IO（onlyconnect） */
    static bool GetOnlyConnect();
    /** 绑核起始 ID（startcoreid） */
    static int64_t GetStartCoreId();
    /** 发送线程数量（numjobs） */
    static int64_t GetNumJobs();
    /** 读 log 的位置（readlogdir） */
    static std::string GetReadLogDir();
    /** 日志输出目录（logdir） */
    static std::string GetLogDir();

    // ---- log 模块参数 ----
    /** 全量日志最低级别（log.level） */
    static spdlog::level::level_enum GetLogLevel();
    /** 日志输出格式（log.pattern），语法同 spdlog pattern，值不能含 '#' */
    static std::string GetLogPattern();
    /** 全量日志实际文件名（log.file_name 的 %Y%m%d_%H 等时间标记已按加载时刻展开） */
    static std::string GetLogFileName();
    /** warn 专用实际文件名（log.warn_file_name，时间标记已展开），记录 warn 及以上 */
    static std::string GetLogWarnFileName();
    /** error 专用实际文件名（log.error_file_name，时间标记已展开），记录 error 及以上 */
    static std::string GetLogErrorFileName();
    /** 单文件大小上限字节（log.max_file_size） */
    static int64_t GetLogMaxFileSize();
    /** 滚动文件保留个数，不含活跃文件（log.max_files），0 表示不清理 */
    static int64_t GetLogMaxFiles();
    /** 异步队列长度（log.async_queue_size） */
    static int64_t GetLogAsyncQueueSize();
    /** 周期刷盘间隔毫秒（log.async_flush_interval_ms） */
    static int64_t GetLogAsyncFlushIntervalMs();

    /** 将全部配置项经 getter 读出并以 info 级别输出到日志；须在 Load 与日志初始化成功后调用 */
    static void Print();

    ConfigUtil() = delete; // 静态工具类，禁止实例化

private:
    /** 配置检查：级别名合法、runtime=0 时 loop 须大于 0；发现问题填 err 并返回 false */
    static bool Check(std::string &err);

    /** 原始键值缓存（全 string；文件名值为加载时刻展开后的实际文件名） */
    inline static std::unordered_map<std::string, std::string> values_;
};
