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

#include "utils/file_util.h"
#include <filesystem>
#include <fstream>
#include <system_error>

bool FileUtil::ReadLines(const std::string &path, std::vector<std::string> &lines, std::string &err)
{
    std::ifstream fin(path);
    if (!fin.is_open()) {
        err = "cannot open file: " + path;
        return false;
    }
    lines.clear();
    std::string line;
    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') { line.pop_back(); }
        lines.emplace_back(line);
    }
    if (fin.bad()) {
        err = "IO error while reading file: " + path;
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
            if (!entry.is_regular_file()) { continue; }
            files.emplace_back(entry.path().filename().string(),
                               static_cast<std::uint64_t>(entry.file_size()));
        }
    } catch (const std::filesystem::filesystem_error &e) {
        err = "failed to list directory: " + dirPath + " (" + e.what() + ")";
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
        err = "failed to create directory: " + dirPath;
        return false;
    }
    return true;
}
