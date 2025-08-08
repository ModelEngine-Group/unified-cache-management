/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UCM_LOCAL_STORE_LOGGER_H
#define UCM_LOCAL_STORE_LOGGER_H

#include <spdlog/async.h>

namespace UCM {

namespace Logger {

std::shared_ptr<spdlog::async_logger> Make();

} // namespace Logger

} // namespace UCM

#define UCM_LOG(level, ...)                                                                                   \
    UCM::Logger::Make()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, level, __VA_ARGS__)
#define UCM_DEBUG(...) UCM_LOG(spdlog::level::debug, __VA_ARGS__)
#define UCM_INFO(...) UCM_LOG(spdlog::level::info, __VA_ARGS__)
#define UCM_WARN(...) UCM_LOG(spdlog::level::warn, __VA_ARGS__)
#define UCM_ERROR(...) UCM_LOG(spdlog::level::err, __VA_ARGS__)

#endif // UCM_LOCAL_STORE_LOGGER_H
