/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "utils/file_util.h"

#include <filesystem>
#include <fstream>
#include <system_error>

bool FileUtil::ReadLines(const std::string &path, std::vector<std::string> &lines,
                         std::string &err)
{
    std::ifstream fin(path);
    if (!fin.is_open()) {
        err = "无法打开文件: " + path;
        return false;
    }
    lines.clear();
    std::string line;
    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.emplace_back(line);
    }
    if (fin.bad()) {
        err = "读取文件发生 IO 错误: " + path;
        return false;
    }
    return true;
}

bool FileUtil::ListFiles(const std::string &dirPath,
                         std::vector<std::pair<std::string, std::uint64_t>> &files,
                         std::string &err)
{
    files.clear();
    try {
        for (const auto &entry : std::filesystem::directory_iterator(dirPath)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            files.emplace_back(entry.path().filename().string(),
                               static_cast<std::uint64_t>(entry.file_size()));
        }
    } catch (const std::filesystem::filesystem_error &e) {
        err = "遍历目录失败: " + dirPath + " (" + e.what() + ")";
        return false;
    }
    return true;
}

bool FileUtil::EnsureDir(const std::string &dirPath, std::string &err)
{
    const std::filesystem::path dir(dirPath);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (!std::filesystem::is_directory(dir, ec)) {
        err = "目录创建失败: " + dirPath;
        return false;
    }
    return true;
}
