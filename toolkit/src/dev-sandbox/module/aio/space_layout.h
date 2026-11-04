/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#ifndef AIO_SPACE_LAYOUT_H
#define AIO_SPACE_LAYOUT_H

#include <string>
#include <sys/stat.h>
#include <vector>
#include "block_type.h"
#include "error_handle.h"

namespace aio {

class SpaceLayout {
public:
    SpaceLayout(std::string root) : root_(std::move(root))
    {
        if (root_.back() != '/') { root_ += '/'; }
        const auto subDirs = GenerateHexStrings(shardSize_);
        for (const auto& dir : subDirs) { MkDir(root_ + dir); }
    }
    std::string BlockPath(const BlockId& block) const
    {
        const auto& file = fmt::format("{}", block);
        const auto& dir = file.substr(0, shardSize_);
        return fmt::format("{}{}/{}", root_, dir, file);
    }

private:
    static std::vector<std::string> GenerateHexStrings(const size_t n)
    {
        if (n == 0) [[unlikely]] { return {}; }
        size_t nCombinations = 1ULL << (n * 4);
        std::vector<std::string> result;
        result.reserve(nCombinations);
        constexpr char hexChars[] = "0123456789abcdef";
        for (size_t i = 0; i < nCombinations; ++i) {
            std::string s(n, '0');
            auto temp = i;
            for (int j = n - 1; j >= 0; --j) {
                s[j] = hexChars[temp & 0xF];
                temp >>= 4;
            }
            result.push_back(s);
        }
        return result;
    }
    static void MkDir(const std::string& dir)
    {
        auto ret = mkdir(dir.c_str(), (S_IRWXU | S_IRWXG | S_IROTH));
        AIO_ASSERT(ret == 0 || errno == EEXIST);
    }

    const size_t shardSize_{3};
    std::string root_;
};

}  // namespace aio

#endif
