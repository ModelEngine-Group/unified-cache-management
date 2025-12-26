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
#ifndef UNIFIEDCACHE_INFRA_LOGGER_SPDLOG_COMPRESS_ROTATE_FILE_SINK_H
#define UNIFIEDCACHE_INFRA_LOGGER_SPDLOG_COMPRESS_ROTATE_FILE_SINK_H

#include <spdlog/details/file_helper.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/details/synchronous_factory.h>
#include <spdlog/sinks/base_sink.h>
#include <mutex>
#include <string>
#include <vector>

namespace spdlog {
namespace sinks {

template <typename Mutex>
class compress_rotate_file_sink final : public base_sink<Mutex> {
public:
    static constexpr size_t MaxFiles = 200000;
    compress_rotate_file_sink(filename_t base_filename,
                       std::size_t max_size,
                       std::size_t max_files,
                       bool rotate_on_open = false,
                       const file_event_handlers &event_handlers = {});
    filename_t filename();
    void rotate_now();
    void set_max_size(std::size_t max_size);
    std::size_t get_max_size();
    void set_max_files(std::size_t max_files);
    std::size_t get_max_files();

protected:
    void sink_it_(const details::log_msg &msg) override;
    void flush_() override;

private:
    bool rotate_(std::string &src_path);
    void compress(const std::string &src_path);
    bool rename_file_(const filename_t &src_filename, const filename_t &target_filename);

    filename_t base_filename_;
    std::size_t max_size_;
    std::size_t max_files_;
    std::size_t current_size_;
    details::file_helper file_helper_;
    std::vector<filename_t> compressed_files_;
};

using compress_rotate_file_sink_mt = compress_rotate_file_sink<std::mutex>;
using compress_rotate_file_sink_st = compress_rotate_file_sink<details::null_mutex>;

}  // namespace sinks

//
// factory functions
//
template <typename Factory = spdlog::synchronous_factory>
std::shared_ptr<logger> compress_rotating_logger_mt(const std::string &logger_name,
                                                    const filename_t &filename,
                                                    size_t max_file_size,
                                                    size_t max_files,
                                                    bool rotate_on_open = false,
                                                    const file_event_handlers &event_handlers = {}) {
    return Factory::template create<sinks::compress_rotate_file_sink_mt>(
        logger_name, filename, max_file_size, max_files, rotate_on_open, event_handlers);
}

template <typename Factory = spdlog::synchronous_factory>
std::shared_ptr<logger> compress_rotating_logger_st(const std::string &logger_name,
                                                    const filename_t &filename,
                                                    size_t max_file_size,
                                                    size_t max_files,
                                                    bool rotate_on_open = false,
                                                    const file_event_handlers &event_handlers = {}) {
    return Factory::template create<sinks::compress_rotate_file_sink_st>(
        logger_name, filename, max_file_size, max_files, rotate_on_open, event_handlers);
}
}  // namespace spdlog

#include "compress_rotate_file-inl.h"

#endif