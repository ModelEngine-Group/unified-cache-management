/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "test/log_test.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "log/logger_manager.h"
#include "utils/config_util.h"
#include "utils/file_util.h"

namespace {

/** 校验日志文件每行级别均在允许集合内（行格式：[时间] [级别] [线程] 内容），输出总行数 */
bool CheckFileLevels(const std::string &path, const std::vector<std::string> &allowed,
                     size_t &lineCount, std::string &err)
{
    std::vector<std::string> lines;
    if (!FileUtil::ReadLines(path, lines, err)) {
        return false;
    }

    lineCount = 0;
    for (const std::string &line : lines) {
        ++lineCount;
        size_t timeBegin = line.find('[');
        size_t levelBegin =
            (timeBegin == std::string::npos) ? std::string::npos : line.find('[', timeBegin + 1);
        size_t levelEnd =
            (levelBegin == std::string::npos) ? std::string::npos : line.find(']', levelBegin);
        if (levelBegin == std::string::npos || levelEnd == std::string::npos) {
            err = "第 " + std::to_string(lineCount) + " 行缺少级别字段";
            return false;
        }

        std::string level = line.substr(levelBegin + 1, levelEnd - levelBegin - 1);
        if (std::find(allowed.begin(), allowed.end(), level) == allowed.end()) {
            err = "第 " + std::to_string(lineCount) + " 行出现不允许的级别 [" + level + "]";
            return false;
        }
    }
    return true;
}

/** 校验单个日志文件并打印结果：可打开、非空（allowEmpty 除外）、行级别合法 */
bool VerifyLogFile(const std::string &path, const std::vector<std::string> &allowed,
                   bool allowEmpty, const std::string &desc)
{
    size_t lineCount = 0;
    std::string err;
    if (!CheckFileLevels(path, allowed, lineCount, err)) {
        std::cout << "[失败] " << desc << "（" << path << "）: " << err << std::endl;
        return false;
    }
    if (lineCount == 0 && !allowEmpty) {
        std::cout << "[失败] " << desc << "（" << path << "）: 文件为空" << std::endl;
        return false;
    }

    std::cout << "[通过] " << desc << "（" << path << "）: " << lineCount << " 行" << std::endl;
    return true;
}

} // namespace

bool RunLogTest()
{
    // 各级别样例；debug 低于配置级别，应被过滤
    LOG_DEBUG("debug 级别日志（应被过滤）");
    LOG_INFO("info 级别日志样例");
    LOG_WARN("warn 级别日志样例：应写入全量日志与 warn 专用日志");
    LOG_ERROR("error 级别日志样例：应写入全量日志、warn 专用日志与 error 专用日志");

    // 批量输出触发全量文件按大小滚动，滚动后再补 warn / error 验证专用文件仍正常
    constexpr int kStressCount = 4000;
    for (int i = 0; i < kStressCount; ++i) {
        LOG_INFO("压测日志 #{:05d}：模拟持续运行输出以触发日志文件按大小滚动", i);
    }
    LOG_WARN("滚动后 warn 级别日志样例");
    LOG_ERROR("滚动后 error 级别日志样例");

    LoggerManager::Shutdown();

    bool allPassed = true;
    std::string err;
    std::filesystem::path dirPath(ConfigUtil::GetLogDir());
    std::cout << "---- 日志文件列表（目录 " << dirPath.string() << "）----" << std::endl;

    // 滚动文件名形如 主名.序号.扩展名（nds_20261112_16.1.log），按展开后的基础名收集全量日志文件
    std::string baseName = ConfigUtil::GetLogFileName();
    std::string baseStem = std::filesystem::path(baseName).stem().string();
    std::string baseExt = std::filesystem::path(baseName).extension().string();
    std::string rollingPrefix = baseStem + ".";
    std::vector<std::string> fullLogFiles;

    std::vector<std::pair<std::string, std::uint64_t>> dirFiles;
    if (!FileUtil::ListFiles(dirPath.string(), dirFiles, err)) {
        std::cout << "[失败] " << err << std::endl;
        return false;
    }
    for (const auto &file : dirFiles) {
        const std::string &name = file.first;
        std::cout << "  " << name << "  " << file.second << " 字节" << std::endl;

        bool isFullLog = (name == baseName);
        if (!isFullLog && name.size() > rollingPrefix.size() + baseExt.size() &&
            name.compare(0, rollingPrefix.size(), rollingPrefix) == 0 &&
            name.compare(name.size() - baseExt.size(), baseExt.size(), baseExt) == 0) {
            std::string index = name.substr(rollingPrefix.size(),
                                            name.size() - rollingPrefix.size() - baseExt.size());
            isFullLog = std::all_of(index.begin(), index.end(), [](char c) {
                return std::isdigit(static_cast<unsigned char>(c)) != 0;
            });
        }
        if (isFullLog) {
            fullLogFiles.push_back((dirPath / name).string());
        }
    }

    // 专用文件级别校验（阈值语义）
    allPassed = VerifyLogFile((dirPath / ConfigUtil::GetLogWarnFileName()).string(),
                              {"warning", "error"}, false,
                              "warn 专用文件仅含 warn 及以上级别") &&
                allPassed;
    allPassed = VerifyLogFile((dirPath / ConfigUtil::GetLogErrorFileName()).string(),
                              {"error"}, false,
                              "error 专用文件仅含 error 级别") &&
                allPassed;

    // 全量文件级别过滤校验（trace / debug 不应出现）
    for (const std::string &file : fullLogFiles) {
        allPassed = VerifyLogFile(file, {"info", "warning", "error"}, false,
                                  "全量文件级别过滤") &&
                    allPassed;
    }

    // 文件数量校验：滚动文件上限 + 1 个活跃文件
    size_t fileLimit = static_cast<size_t>(ConfigUtil::GetLogMaxFiles()) + 1;
    if (fullLogFiles.size() > fileLimit) {
        std::cout << "[失败] 全量日志文件数 " << fullLogFiles.size() << " 超过上限 " << fileLimit
                  << std::endl;
        allPassed = false;
    } else {
        std::cout << "[通过] 全量日志文件数 " << fullLogFiles.size() << "（上限 " << fileLimit
                  << "）" << std::endl;
    }

    std::cout << (allPassed ? "---- 日志功能验证通过 ----" : "---- 日志功能验证失败 ----")
              << std::endl;
    return allPassed;
}
