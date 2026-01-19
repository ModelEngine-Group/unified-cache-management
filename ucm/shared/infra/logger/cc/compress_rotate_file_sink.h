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
#ifndef UNIFIEDCACHE_INFRA_LOGGER_SPDLOG_rotating_file_sink_H
#define UNIFIEDCACHE_INFRA_LOGGER_SPDLOG_rotating_file_sink_H

#include <chrono>
#include <fstream>
#include <mutex>
#include <spdlog/details/file_helper.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/details/os.h>
#include <spdlog/details/synchronous_factory.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <string>
#include <vector>
#include <zlib.h>

namespace spdlog {
namespace sinks {

template <>
class rotating_file_sink<std::mutex> final : public base_sink<std::mutex> {
public:
    static constexpr size_t MaxFiles = 200000;
    SPDLOG_INLINE rotating_file_sink(filename_t base_filename, std::size_t max_size,
                                     std::size_t max_files, bool rotate_on_open = false,
                                     const file_event_handlers &event_handlers = {})
        : base_filename_(std::move(base_filename)),
          max_size_(max_size),
          max_files_(max_files),
          file_helper_{event_handlers}
    {
        if (max_size == 0) {
            throw_spdlog_ex("rotating sink constructor: max_size arg cannot be zero");
        }

        if (max_files > MaxFiles) {
            throw_spdlog_ex("rotating sink constructor: max_files arg cannot exceed MaxFiles");
        }
        file_helper_.open(base_filename_);
        current_size_ = file_helper_.size();  // expensive. called only once
        if (rotate_on_open && current_size_ > 0) {
            std::string src_path = "";
            if (!rotate_(src_path)) {
                throw_spdlog_ex("rotating_file_sink: failed rotating file" + base_filename_);
            }
            if (!compress_(src_path)) {
                throw_spdlog_ex("rotating_file_sink: failed compressing file" + base_filename_);
            }
            current_size_ = 0;
        }
    }

private:
    SPDLOG_INLINE void sink_it_(const details::log_msg &msg)
    {
        memory_buf_t formatted;
        base_sink<std::mutex>::formatter_->format(msg, formatted);
        auto new_size = current_size_ + formatted.size();

        if (new_size > max_size_) {
            file_helper_.flush();
            if (file_helper_.size() > 0) {
                std::string src_path = "";
                if (!rotate_(src_path)) {
                    throw_spdlog_ex("rotating_file_sink: failed rotating file" + base_filename_);
                }
                if (!compress_(src_path)) {
                    throw_spdlog_ex("rotating_file_sink: failed compressing file" + base_filename_);
                }
                new_size = formatted.size();
            }
        }
        file_helper_.write(formatted);
        current_size_ = new_size;
    }

    SPDLOG_INLINE void flush_() { file_helper_.flush(); }

    SPDLOG_INLINE bool rotate_(std::string &src_path)
    {
        using details::os::filename_to_str;
        using details::os::path_exists;

        filename_t current_file = base_filename_;
        file_helper_.close();

        if (path_exists(current_file)) {
            filename_t target;
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            auto ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) %
                1000;

            std::tm *tm = std::localtime(&time_t);
            char time_str[64];
            std::strftime(time_str, sizeof(time_str), "%Y-%m-%d_%H-%M-%S", tm);

            char time_str_with_ms[80];
            std::snprintf(time_str_with_ms, sizeof(time_str_with_ms), "%s-%03lld", time_str,
                          static_cast<long long>(ms.count()));

            filename_t basename;
            filename_t ext;
            std::tie(basename, ext) = details::file_helper::split_by_extension(base_filename_);
            target = fmt_lib::format(SPDLOG_FMT_STRING(SPDLOG_FILENAME_T("{}_{}{}")), basename,
                                     time_str_with_ms, ext);
            src_path = target;

            if (!rename_file_(current_file, target)) {
                file_helper_.reopen(true);
                current_size_ = 0;
                throw_spdlog_ex("rotating_file_sink: failed renaming " +
                                    filename_to_str(current_file) + " to " +
                                    filename_to_str(target),
                                errno);
                return false;
            }
            compressed_files_.push_back(target);
        }

        while (compressed_files_.size() > max_files_) {
            filename_t target_file = compressed_files_.front() + ".gz";
            if (path_exists(target_file)) {
                (void)details::os::remove(target_file);
                compressed_files_.erase(compressed_files_.begin());
            }
        }
        file_helper_.reopen(true);
        return true;
    }

    SPDLOG_INLINE bool compress_(const std::string &src_path)
    {
        if (max_files_ == 0) { return true; }
        const std::string dest_path = src_path + ".gz";
        std::ifstream src_file(src_path, std::ios::binary);
        if (!src_file.is_open()) { throw_spdlog_ex("Error: Could not open source file."); }

        gzFile out_file = gzopen(dest_path.c_str(), "wb");
        if (!out_file) { throw_spdlog_ex("Error: Could not open destination file."); }

        const size_t buffer_size = 16384;  // 16 KB chunks
        std::vector<char> buffer(buffer_size);

        while (src_file.read(buffer.data(), buffer_size) || src_file.gcount() > 0) {
            int bytes_read = static_cast<int>(src_file.gcount());
            if (gzwrite(out_file, buffer.data(), bytes_read) <= 0) {
                throw_spdlog_ex("Error writing compressed data.");
                gzclose(out_file);
                return false;
            }
        }

        gzclose(out_file);
        src_file.close();

        std::remove(src_path.c_str());
        return true;
    }

    SPDLOG_INLINE bool rename_file_(const filename_t &src_filename,
                                    const filename_t &target_filename)
    {
        (void)details::os::remove(target_filename);
        return details::os::rename(src_filename, target_filename) == 0;
    }

    filename_t base_filename_;
    std::size_t max_size_;
    std::size_t max_files_;
    std::size_t current_size_;
    details::file_helper file_helper_;
    std::vector<filename_t> compressed_files_;
};

using rotating_file_sink_mt = rotating_file_sink<std::mutex>;
using rotating_file_sink_st = rotating_file_sink<details::null_mutex>;

}  // namespace sinks

//
// factory functions
//
template <typename Factory = spdlog::synchronous_factory>
std::shared_ptr<logger> compress_rotating_logger_mt(const std::string &logger_name,
                                                    const filename_t &filename,
                                                    size_t max_file_size, size_t max_files,
                                                    bool rotate_on_open = false,
                                                    const file_event_handlers &event_handlers = {})
{
    return Factory::template create<sinks::rotating_file_sink_mt>(
        logger_name, filename, max_file_size, max_files, rotate_on_open, event_handlers);
}

template <typename Factory = spdlog::synchronous_factory>
std::shared_ptr<logger> compress_rotating_logger_st(const std::string &logger_name,
                                                    const filename_t &filename,
                                                    size_t max_file_size, size_t max_files,
                                                    bool rotate_on_open = false,
                                                    const file_event_handlers &event_handlers = {})
{
    return Factory::template create<sinks::rotating_file_sink_st>(
        logger_name, filename, max_file_size, max_files, rotate_on_open, event_handlers);
}
}  // namespace spdlog

#endif