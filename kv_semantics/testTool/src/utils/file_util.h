/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

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

    FileUtil() = delete; // 静态工具类，禁止实例化
};
