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
#include <utility>
#include <vector>

/**
 * 文件工具集：集中管理文件的打开/关闭/读取与目录操作，
 * 业务代码不直接接触 ifstream / filesystem 细节。
 */
class FileUtil {
public:
    /** 读取整个文件为行列表（剥离行尾 \r；空文件成功且 lines 为空） */
    static bool ReadLines(const std::string &path, std::vector<std::string> &lines,
                          std::string &err);

    /** 列出目录下全部常规文件的名称与字节大小（不含子目录；条目顺序不保证） */
    static bool ListFiles(const std::string &dirPath,
                          std::vector<std::pair<std::string, std::uint64_t>> &files,
                          std::string &err);

    /** 确保目录存在（不存在则递归创建；已存在且为目录则直接成功） */
    static bool EnsureDir(const std::string &dirPath, std::string &err);

    FileUtil() = delete;  // 静态工具类，禁止实例化
};
