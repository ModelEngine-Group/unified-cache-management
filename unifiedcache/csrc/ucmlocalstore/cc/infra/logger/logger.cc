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
#include "logger.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/details/os.h>

#define UCM_LOCAL_STORE_LOGGER_NAME "UCMLOCALSTORE"
#define UCM_LOCAL_STORE_ENV_LOGGER_LEVEL "UCMLOCALSTORE_LOGGER_LEVEL"

namespace UCM {

namespace Logger {

std::shared_ptr<spdlog::async_logger> Make()
{
    static auto pool = std::make_shared<spdlog::details::thread_pool>(8192, 1);
    static auto logger = [] {
        auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

        auto logger = std::make_shared<spdlog::async_logger>(
            UCM_LOCAL_STORE_LOGGER_NAME,
            sink,
            pool,
            spdlog::async_overflow_policy::block
        );

        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f %z] [%n] [%^%L%$] %v [%P,%t] [%s:%#,%!]");

        auto level = spdlog::details::os::getenv(UCM_LOCAL_STORE_ENV_LOGGER_LEVEL);
        logger->set_level(level.empty() ? spdlog::level::info : spdlog::level::from_str(level));

        return logger;
    }();

    return logger;
}

} // namespace Logger

} // namespace UCM
