/**
/* MIT License
/*
/* Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
/*
/* Permission is hereby granted, free of charge, to any person obtaining a copy
/* of this software and associated documentation files (the "Software"), to deal
/* in the Software without restriction, including without limitation the rights
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/* copies of the Software, and to permit persons to whom the Software is
/* furnished to do so, subject to the following conditions:
/*
/* The above copyright notice and this permission notice shall be included in all
/* copies or substantial portions of the Software.
/*
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/* SOFTWARE.
 * */
#include "logger.h"
#include <spdlog/cfg/helpers.h>
#include <spdlog/details/os.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace UC {

static std::mutex g_mutex;
static std::shared_ptr<spdlog::logger> g_logger = nullptr;

std::shared_ptr<spdlog::logger> Logger::Make()
{
    if (g_logger) { return g_logger; }
    std::unique_lock lock(g_mutex);
    if (g_logger) { return g_logger; }
    try {
        const std::string name = "UCMNFSSTORE";
        const std::string envLevel = name + "_LOGGER_LEVEL";
        g_logger = spdlog::stdout_color_mt(name);
        g_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f %z] [%n] [%^%L%$] %v [%P,%t] [%s:%#,%!]");
        auto level = spdlog::details::os::getenv(envLevel.c_str());
        if (!level.empty()) { spdlog::cfg::helpers::load_levels(level); }
        return g_logger;
    } catch (...) {
        return spdlog::default_logger();
    }
}

} // namespace UC
