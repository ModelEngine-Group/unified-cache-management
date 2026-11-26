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
#include <string>
#include <unordered_map>
#include "utils/config_parse.h"
#include "utils/log_config_util.h"

/**
 * 全局配置中心（单例）：加载配置文件，原始键值以 string 存入缓存，getter 按需解析为
 * 目标类型；缺 key / 坏值抛 std::runtime_error（由 main 统一 catch）；Load/Check 保持
 * bool + err 契约；日志项（log.* 键）经 LogConfigUtil 独立解析管理。
 * 需要读配置的组件在初始化入口显式传入实例引用，不在内部直接访问单例。
 */
class ConfigUtil {
public:
    static ConfigUtil &GetInstance();

    /** 加载配置（解析 key=value、剥离行内注释、日志项解析、Check 校验）；
     * 任一环节失败则整体失败 */
    bool Load(const std::string &path, std::string &err);

    // ---- 测试通用参数 ----
    int GetTestMode() const { return config_parse::RequireInt(values_, "testmode"); }
    int GetRuntime() const { return config_parse::RequireInt(values_, "runtime"); }
    int GetLoop() const { return config_parse::RequireInt(values_, "loop"); }
    bool GetPerf() const { return config_parse::RequireValue(values_, "perf") == "on"; }
    bool GetOnlyConnect() const
    {
        return config_parse::RequireValue(values_, "onlyconnect") == "on";
    }
    int GetAutoDisconnect() const { return config_parse::RequireInt(values_, "autodisconnect"); }
    int GetStartCoreId() const { return config_parse::RequireInt(values_, "startcoreid"); }
    int GetNumJobs() const { return config_parse::RequireInt(values_, "numjobs"); }
    int64_t GetBufferSize() const { return config_parse::RequireInt64(values_, "buffersize"); }
    int GetBufferCount() const { return config_parse::RequireInt(values_, "buffercount"); }
    std::string GetBufferType() const { return config_parse::RequireValue(values_, "buffertype"); }
    int GetDeviceId() const { return config_parse::RequireInt(values_, "deviceid"); }
    const LogConfigUtil &GetLogConfig() const { return logConfig_; }

    void Print() const;

    ConfigUtil(const ConfigUtil &) = delete;             // 单例：禁止拷贝
    ConfigUtil &operator=(const ConfigUtil &) = delete;  // 单例：禁止赋值

private:
    ConfigUtil() = default;  // 单例：私有默认构造，仅 GetInstance 内部创建

    bool Check(std::string &err);

    /** 原始键值缓存（全 string；不含日志键） */
    std::unordered_map<std::string, std::string> values_;

    /** 日志配置模块（log.* 键的解析结果） */
    LogConfigUtil logConfig_;
};
