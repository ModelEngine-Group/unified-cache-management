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
#ifndef UNIFIEDCACHE_LOGGER_H
#define UNIFIEDCACHE_LOGGER_H

#include <spdlog/spdlog.h>

namespace UC {

class Logger {
public:
    static std::shared_ptr<spdlog::logger> Make();
};

} // namespace UC

#define UC_LOG(level, ...)                                                                                             \
    UC::Logger::Make()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, level, __VA_ARGS__)
#define UC_DEBUG(...) UC_LOG(spdlog::level::debug, __VA_ARGS__)
#define UC_INFO(...) UC_LOG(spdlog::level::info, __VA_ARGS__)
#define UC_WARN(...) UC_LOG(spdlog::level::warn, __VA_ARGS__)
#define UC_ERROR(...) UC_LOG(spdlog::level::err, __VA_ARGS__)

#endif
