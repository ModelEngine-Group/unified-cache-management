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
#include <vector>
#include "logger/logger.h"

struct UserType {
    int32_t value;
    explicit UserType(int32_t v) : value(v) {}
};

template <>
struct fmt::formatter<UserType> : fmt::formatter<std::string> {
    auto format(UserType my, format_context& ctx) const -> decltype(ctx.out())
    {
        return fmt::format_to(ctx.out(), "[UserType value={}]", my.value);
    }
};

int main(int32_t argc, char const* argv[])
{
    LOG_INFO("Running with {} arguments.", argc);
    for (auto i = 0; i < argc; ++i) { LOG_DEBUG("Argument {}: {}.", i, argv[i]); }
    LOG_WARN("Easy padding in numbers like {:08d}.", 12);
    LOG_CRITICAL("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}.", 42);
    LOG_INFO("Support for floats {:03.2f}.", 1.23456);
    LOG_INFO("Positional args are {1} {0}..", "too", "supported");
    LOG_INFO("{:>8} aligned, {:<8} aligned.", "right", "left");
    std::vector<int> vec = {1, 2, 3};
    LOG_INFO("Vector example: {}.", vec);
    LOG_ERROR("Custom type example: {}.", UserType(42));
    return 0;
}
