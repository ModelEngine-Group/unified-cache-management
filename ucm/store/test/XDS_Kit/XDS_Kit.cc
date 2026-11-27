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
#include <cctype>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <list>
#include <mutex>
#include <pthread.h>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
#include "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include/acl/acl.h"
#include "nds_api.h"

#define LOG_ERROR(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)
#define NDS_MAJOR_VERSION_1
#define ALIGNMENT 4096  // 4K块对齐
#define SIZE_1K (1UL << 10)
#define SIZE_8K (1UL << 13)
#define SIZE_512K (1UL << 19)
#define SIZE_1M (1UL << 20)
#define SIZE_16M (1UL << 24)
#define SIZE_1G (1UL << 30)
#define BATCH_NR (64 * 1024)

#define OFFSET_RATIO 4

#define UUID_LEN 64
int repeattimes = 100;

void* devPtr_base;
int device_id = 0;
std::string file_path;
std::string file_name;
off_t start_offset = 0;
off_t ptr_offset = 0;
std::string io_size = "1M";
std::string file_size = "100M";
std::string io_size_arl;
int xfer_type = 1;
int transfer_type = 0;
int random_seed = 0;
ssize_t file_byte_size = SIZE_16M;
ssize_t buf_size = SIZE_1M;
ssize_t buf_size_arl = SIZE_1M;
int application_page_size = 1;
int number_runs = 0;

struct Time_se_ns {
    long Sec;
    long Ns;
};

ssize_t parse_size(const std::string& io_size)
{
    if (io_size.empty()) {
        LOG_ERROR("-s or -i : string is empty\n");
        return -1;
    }
    size_t pos = 0;
    while (pos < io_size.length() && std::isdigit(io_size[pos])) { pos++; }
    if (pos == 0) {
        LOG_ERROR("-s or -i : invalid size format, no digits found in '%s'\n", io_size.c_str());
        return -1;
    }

    std::string num_str = io_size.substr(0, pos);
    unsigned long long value = std::stoull(num_str);
    ssize_t multiplier = 1;
    if (pos < io_size.length()) {
        char unit = std::toupper(io_size[pos]);
        switch (unit) {
            case 'K': multiplier = SIZE_1K; break;
            case 'M': multiplier = SIZE_1M; break;
            case 'G': multiplier = SIZE_1G; break;
            default:
                LOG_ERROR(
                    "-s or -i : invalid size unit '%c' in '%s', supported units are K, M, G\n",
                    unit, io_size.c_str());
                return -1;
        }
        if (pos + 1 != io_size.length()) {
            LOG_ERROR("-s or -i : invalid size format '%s', extra characters after unit\n",
                      io_size.c_str());
            return -1;
        }
    }
    unsigned long long result = value * multiplier;
    if (result > static_cast<unsigned long long>(SSIZE_MAX)) {
        LOG_ERROR("-s or -i : size value '%s' exceeds maximum allowed value\n", io_size.c_str());
        return -1;
    }
    return static_cast<ssize_t>(result);
}

off_t get_file_size(int fd)
{
    off_t size = lseek(fd, 0, SEEK_END);
    if (size != -1) { lseek(fd, 0, SEEK_SET); }
    return size;
}

Time_se_ns timecont(long* sec_input, long* ns_input)
{
    Time_se_ns ret;
    long cur_sec = 0;
    long cur_ns = 0;

    for (int i = 0; i < repeattimes; ++i) {
        cur_sec += sec_input[i];
        cur_ns += ns_input[i] / repeattimes;  // 避免溢出
    }

    ret.Sec = cur_sec / repeattimes;  // 秒数取整
    long sec_to_ns = (cur_sec % repeattimes) * (1E+09) / repeattimes;

    cur_ns += sec_to_ns;
    ret.Ns = cur_ns;
    return ret;
}

void* alig_malloc(int64_t buf)
{
    int64_t ret = 0;
    int current_device = -1;
    void* ptr;

    ret = aclrtSetDevice(device_id);
    if (ret) {
        LOG_ERROR("aclrtSetDevice failed, nds_driver initialization failed, ret:%d\n", ret);
        return NULL;
    }

    ret = aclrtGetDevice(&current_device);
    if (application_page_size == 1) {
        ret = aclrtMalloc(&ptr, buf, ACL_MEM_MALLOC_HUGE_ONLY);
    } else {
        ret = aclrtMalloc(&ptr, buf, ACL_MEM_MALLOC_NORMAL_ONLY);
    }
    if (ret) {
        LOG_ERROR("Allocated memory address failed: %p\n", ptr);
        return NULL;
    }
    void* aligned_ptr = (void*)((unsigned long long)ptr);
    return aligned_ptr;
}

int calculate_max_offset(ssize_t file_size)
{
    int max_offset = file_size - buf_size;
    return max_offset;
}

void alig_free(void* aligned_ptr) { aclrtFree(aligned_ptr); }

int buf_check(void* devPtr_base, ssize_t buf_size, off_t devPtr_offset, off_t file_offset)
{
    // show HBM data
    char* check_buff = (char*)malloc(buf_size);
    aclrtMemcpy(check_buff, buf_size, devPtr_base, buf_size, ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 26; i++) { printf("%c", check_buff[i]); }
    printf("\n");

    for (int i = 0; i < buf_size; i++) {
        if (check_buff[i] != 'a' + (i + file_offset) % 26) {
            LOG_ERROR("\nread data check wrong!");
            free(check_buff);
            return -1;
        }
    }
    free(check_buff);
    return 0;
}

long bd_compute(long sec, long ns, int times, size_t size)
{
    return ((size * times / (SIZE_1K * SIZE_1K)) * 1000000000) / ((sec) * 1000000000 + (ns));
}

int test_band_dtn_write(int thread_num, ssize_t buf_size, int run_time_input)
{
    int times = 0;
    int run_time = run_time_input;

    void* devPtr_base = alig_malloc(buf_size_arl);
    char* write_buff = (char*)malloc(buf_size);
    long bandwidth;
    std::string filename;
    for (size_t i = 0; i < buf_size; ++i) { write_buff[i] = 'a' + (i % 26); }
    aclrtMemcpy(devPtr_base, buf_size, write_buff, buf_size, ACL_MEMCPY_HOST_TO_DEVICE);

    if (!file_name.empty()) {
        filename = file_name;
    } else {
        filename = file_path + "/file" + std::to_string(thread_num) + ".txt";
    }
    std::cout << "open " << filename << std::endl;

    // 检查文件是否存在，不存在则创建并初始化
    struct stat file_stat;
    if (stat(filename.c_str(), &file_stat) != 0) {
        // 文件不存在，创建并初始化
        int fd_init = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd_init < 0) {
            LOG_ERROR("file create %s errno\n", filename.c_str());
            alig_free(devPtr_base);
            free(write_buff);
            return -1;
        }
        if (posix_fallocate(fd_init, 0, file_byte_size) != 0) {
            perror("posix_fallocate failed");
            close(fd_init);
            alig_free(devPtr_base);
            free(write_buff);
            return -1;
        }
        close(fd_init);
    }
    int fd = open(filename.c_str(), O_WRONLY | O_DIRECT, 0644);
    if (fd < 0) {
        LOG_ERROR("file open %s errno\n", filename.c_str());
        alig_free(devPtr_base);
        free(write_buff);
        return -1;
    }

    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_BOOTTIME, &start);
    off_t fd_file_size = get_file_size(fd);
    if (fd_file_size < 0) {
        LOG_ERROR("failed to get file size for %s\n", filename.c_str());
        close(fd);
        alig_free(devPtr_base);
        free(write_buff);
        return -1;
    }
    int max_offset = calculate_max_offset(fd_file_size);
    off_t random_offset = 0;
    srand(time(NULL));
    auto start_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();

    do {
        if (start_offset != 0) {
            random_offset = start_offset;
        } else if (transfer_type == 3) {
            random_offset = rand() % max_offset;
        } else {
            random_offset = 0;
        }
        lseek(fd, random_offset, SEEK_SET);
        if (xfer_type == 1) {
            aclrtMemcpy(devPtr_base, buf_size, write_buff, buf_size, ACL_MEMCPY_DEVICE_TO_HOST);
        }
        int ret = write(fd, write_buff, buf_size);
        if (ret < 0 || ret != buf_size) {
            LOG_ERROR(
                "ret: %d, process: %d, times : %d, dtn_write failed, size:%ldk, "
                "bytes_written:%ldk\n",
                ret, device_id, times, buf_size / SIZE_1K, ret / SIZE_1K);
            break;
        }
        if (number_runs == 1) {
            printf("End of single run.");
            break;
        }
        // 确保数据写入磁盘
        // fsync(fd);
        if (((++times) % 1000) == 0) {
            clock_gettime(CLOCK_BOOTTIME, &end);
            bandwidth =
                bd_compute(end.tv_sec - start.tv_sec, end.tv_nsec - start.tv_nsec, 1000, buf_size);
            printf(
                "process: deviceid: %d, thread: %d, write %ldM by bandwidth: %ld M/s, test rounds: "
                "%d\n",
                device_id, thread_num, buf_size / (SIZE_1K * SIZE_1K), bandwidth, times);
            clock_gettime(CLOCK_BOOTTIME, &start);
        }
        current_time = std::chrono::steady_clock::now();
        elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
    } while (elapsed < run_time);
    int avg_bandwidth = buf_size * times / (SIZE_1K * SIZE_1K) / run_time;
    int final_iops = times / run_time;
    double final_avg_latency = (double)run_time * 1000000L / times;  // 转换为微秒
    printf(
        "__________process: deviceid: %d, thread: %d end, return %ld M/s, IOPS: %d, Avg Latency: "
        "%.2f us________\n",
        device_id, thread_num, avg_bandwidth, final_iops, final_avg_latency);
    alig_free(devPtr_base);
    free(write_buff);
    close(fd);
    return avg_bandwidth;
}

long test_band_nds_write(int thread_num, ssize_t buf_size, int run_time_input)
{
    int times = 0;
    int run_time = run_time_input;
    void* devPtr_base = alig_malloc(buf_size_arl);
    char* write_buff = (char*)malloc(buf_size);
    for (size_t i = 0; i < buf_size; ++i) { write_buff[i] = 'a' + (i % 26); }
    aclrtMemcpy(devPtr_base, buf_size, write_buff, buf_size, ACL_MEMCPY_HOST_TO_DEVICE);
    free(write_buff);
    struct timespec start;
    struct timespec end;
    long bandwidth;
    long avg_bandwidth;
    std::string filename;
    NdsFileDescr_t descr;
    NdsFileHandle_t fh;
    if (!file_name.empty()) {
        filename = file_name;
    } else {
        filename = file_path + "/file" + std::to_string(thread_num) + ".txt";
    }
    std::cout << "open " << filename << std::endl;
    struct stat file_stat;
    if (stat(filename.c_str(), &file_stat) != 0) {
        int fd_init = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd_init < 0) {
            LOG_ERROR("file create %s errno\n", filename.c_str());
            alig_free(devPtr_base);
            return -1;
        }
        if (posix_fallocate(fd_init, 0, file_byte_size) != 0) {
            perror("posix_fallocate failed");
            close(fd_init);
            alig_free(devPtr_base);
            return -1;
        }
        close(fd_init);
    }
    int fd = open(filename.c_str(), O_WRONLY | O_DIRECT, 0644);
    if (fd < 0) {
        LOG_ERROR("file open %s errno\n", filename);
        alig_free(devPtr_base);
        return -1;
    }
    off_t fd_file_size = get_file_size(fd);
    if (fd_file_size < 0) {
        LOG_ERROR("failed to get file size for %s\n", filename.c_str());
        close(fd);
        alig_free(devPtr_base);
        return -1;
    }
    memset(&descr, 0, sizeof(descr));
    descr.fd = fd;
    NdsFileError_t status = NdsFileHandleRegister(&fh, &descr);
    if (status.err != NDS_FILE_SUCCESS) {
        LOG_ERROR("NdsFileHandleRegister failed, ret(%d).", status.err);
        close(fd);
        alig_free(devPtr_base);
        return -1;
    }
    int max_offset = calculate_max_offset(fd_file_size);
    off_t random_offset = 0;
    srand(time(NULL));
    clock_gettime(CLOCK_BOOTTIME, &start);
    auto start_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
    do {
        if (start_offset != 0) {
            random_offset = start_offset;
        } else if (transfer_type == 3) {
            random_offset = rand() % max_offset;
        } else {
            random_offset = 0;
        }
        int ret = NdsFileWrite(fh, devPtr_base, buf_size, random_offset, ptr_offset);
        if (ret < 0 || ret != buf_size) {
            LOG_ERROR("process: %d, times : %d, nds_write failed, size:%ldk, bytes_written:%ldk\n",
                      device_id, times, buf_size / SIZE_1K, ret / SIZE_1K);
            break;
        }
        if (number_runs == 1) {
            printf("End of single run.");
            break;
        }
        // fsync(fd);

        if (((++times) % 1000) == 0) {
            clock_gettime(CLOCK_BOOTTIME, &end);
            bandwidth =
                bd_compute(end.tv_sec - start.tv_sec, end.tv_nsec - start.tv_nsec, 1000, buf_size);
            printf(
                "process: deviceid: %d, thread: %d, write %ldM by bandwidth: %ld M/s, test rounds: "
                "%d\n",
                device_id, thread_num, buf_size / (SIZE_1K * SIZE_1K), bandwidth, times);
            clock_gettime(CLOCK_BOOTTIME, &start);
        }

        current_time = std::chrono::steady_clock::now();
        elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
    } while (elapsed < run_time);
    alig_free(devPtr_base);
    avg_bandwidth = buf_size * times / (SIZE_1K * SIZE_1K) / run_time;
    int final_iops = times / run_time;
    double final_avg_latency = (double)run_time * 1000000L / times;  // 转换为微秒
    printf(
        "__________process: deviceid: %d, thread: %d end, return %ld M/s, IOPS: %d, Avg Latency: "
        "%.2f us________\n",
        device_id, thread_num, avg_bandwidth, final_iops, final_avg_latency);
    NdsFileHandleDeregister(fh);
    close(fd);
    return avg_bandwidth;
}

int test_band_dtn_read(int thread_num, ssize_t buf_size, int run_time_input)
{
    int times = 0;
    int run_time = run_time_input;

    void* devPtr_base = alig_malloc(buf_size_arl);
    char* read_buff = (char*)malloc(buf_size);
    long bandwidth;
    long avg_bandwidth;
    std::string filename;
    if (!file_name.empty()) {
        filename = file_name;
    } else {
        filename = file_path + "/file" + std::to_string(thread_num) + ".txt";
    }
    std::cout << "open " << filename << std::endl;
    int fd = open(filename.c_str(), O_RDONLY | O_DIRECT, 0644);
    if (fd < 0) {
        LOG_ERROR("file open %s errno\n", filename);
        free(read_buff);
        alig_free(devPtr_base);
        return -1;
    }
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_BOOTTIME, &start);
    int round = 0;
    off_t fd_file_size = get_file_size(fd);
    if (fd_file_size < 0) {
        LOG_ERROR("failed to get file size for %s\n", filename.c_str());
        free(read_buff);
        close(fd);
        alig_free(devPtr_base);
        return -1;
    }
    int max_offset = calculate_max_offset(fd_file_size);
    off_t random_offset = 0;
    srand(time(NULL));
    auto start_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
    do {
        if (start_offset != 0) {
            random_offset = start_offset;
        } else if (transfer_type == 2) {
            random_offset = rand() % max_offset;
        } else {
            random_offset = 0;
        }
        lseek(fd, random_offset, SEEK_SET);
        int ret = read(fd, read_buff, buf_size);
        if (ret < 0 || ret != buf_size) {
            LOG_ERROR(
                "ret: %ld, process: %d, times : %d, dtn_read failed, size:%ldk, bytes_read:%ldk\n",
                ret, device_id, times, buf_size / SIZE_1K, ret / SIZE_1K);
            break;
        }
        if (xfer_type == 1) {
            aclrtMemcpy(devPtr_base, buf_size, read_buff, buf_size, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (number_runs == 1) {
            printf("End of single run.");
            break;
        }

        if (((++times) % 1000) == 0) {
            clock_gettime(CLOCK_BOOTTIME, &end);
            bandwidth =
                bd_compute(end.tv_sec - start.tv_sec, end.tv_nsec - start.tv_nsec, 1000, buf_size);
            printf(
                "process: deviceid: %d, thread: %d, read %ldM by bandwidth: %ld M/s, test rounds: "
                "%d\n",
                device_id, thread_num, buf_size / (SIZE_1K * SIZE_1K), bandwidth, times);
            clock_gettime(CLOCK_BOOTTIME, &start);
        }
        current_time = std::chrono::steady_clock::now();
        elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
    } while (elapsed < run_time);
    if (xfer_type != 2) { buf_check(devPtr_base, buf_size, random_offset, random_offset); }
    avg_bandwidth = buf_size * times / (SIZE_1K * SIZE_1K) / run_time;
    int final_iops = times / run_time;
    double final_avg_latency = (double)run_time * 1000000L / times;  // 转换为微秒
    printf(
        "__________process: deviceid: %d, thread: %d end, return %ld M/s, IOPS: %d, Avg Latency: "
        "%.2f us________\n",
        device_id, thread_num, avg_bandwidth, final_iops, final_avg_latency);
    alig_free(devPtr_base);
    close(fd);
    free(read_buff);
    return avg_bandwidth;
}

long test_band_nds_read(int thread_num, ssize_t buf_size, int run_time_input)
{
    int count_band = 0;
    long total_band = 0;
    int times = 0;
    int run_time = run_time_input;
    void* devPtr_base = alig_malloc(buf_size_arl);
    struct timespec start;
    struct timespec end;
    long bandwidth;
    long avg_bandwidth;
    std::string filename;
    NdsFileDescr_t descr;
    NdsFileHandle_t fh;
    if (!file_name.empty()) {
        filename = file_name;
    } else {
        filename = file_path + "/file" + std::to_string(thread_num) + ".txt";
    }
    std::cout << "open " << filename << std::endl;
    int fd = open(filename.c_str(), O_RDONLY | O_DIRECT, 0644);
    if (fd < 0) {
        LOG_ERROR("file open %s errno\n", filename);
        alig_free(devPtr_base);
        return -1;
    }
    off_t fd_file_size = get_file_size(fd);
    if (fd_file_size < 0) {
        LOG_ERROR("failed to get file size for %s\n", filename.c_str());
        close(fd);
        alig_free(devPtr_base);
        return -1;
    }
    memset(&descr, 0, sizeof(descr));
    descr.fd = fd;
    NdsFileError_t status = NdsFileHandleRegister(&fh, &descr);
    if (status.err != NDS_FILE_SUCCESS) {
        LOG_ERROR("NdsFileHandleRegister failed, ret(%d).", status.err);
        close(fd);
        alig_free(devPtr_base);
        return -1;
    }
    int max_offset = calculate_max_offset(fd_file_size);
    off_t random_offset = 0;
    srand(time(NULL));
    clock_gettime(CLOCK_BOOTTIME, &start);
    auto start_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
    do {
        if (start_offset != 0) {
            random_offset = start_offset;
        } else if (transfer_type == 2) {
            random_offset = rand() % max_offset;
        } else {
            random_offset = 0;
        }
        int ret = NdsFileRead(fh, devPtr_base, buf_size, random_offset, ptr_offset);
        if (ret < 0 || ret != buf_size) {
            LOG_ERROR("process: %d, times : %d, nds_read failed, size:%ldk, bytes_read:%ldk\n",
                      device_id, times, buf_size / SIZE_1K, ret / SIZE_1K);
            break;
        }
        if (number_runs == 1) {
            printf("End of single run.");
            break;
        }
        if (((++times) % 1000) == 0) {
            clock_gettime(CLOCK_BOOTTIME, &end);
            bandwidth =
                bd_compute(end.tv_sec - start.tv_sec, end.tv_nsec - start.tv_nsec, 1000, buf_size);
            printf(
                "process: deviceid: %d, thread: %d, read %ldM by bandwidth: %ld M/s, test rounds: "
                "%d\n",
                device_id, thread_num, buf_size / (SIZE_1K * SIZE_1K), bandwidth, times);
            clock_gettime(CLOCK_BOOTTIME, &start);
        }
        current_time = std::chrono::steady_clock::now();
        elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
    } while (elapsed < run_time);
    if (xfer_type != 2) { buf_check(devPtr_base, buf_size, random_offset, random_offset); }
    NdsFileHandleDeregister(fh);
    alig_free(devPtr_base);
    close(fd);
    avg_bandwidth = buf_size * times / (SIZE_1K * SIZE_1K) / run_time;
    int final_iops = times / run_time;
    double final_avg_latency = (double)run_time * 1000000L / times;  // 转换为微秒
    printf(
        "__________process: deviceid: %d, thread: %d end, return %ld M/s, IOPS: %d, Avg Latency: "
        "%.2f us________\n",
        device_id, thread_num, avg_bandwidth, final_iops, final_avg_latency);
    return avg_bandwidth;
}

long test_band_dtn_write_multi(ssize_t buf_size, int run_time_input, int thread_num)
{
    if (thread_num <= 1) {
        return test_band_dtn_write(0, buf_size, run_time_input);
    } else {
        std::vector<std::thread> threads;
        std::vector<long> results(thread_num);
        for (int i = 0; i < thread_num; ++i) {
            threads.emplace_back([i, buf_size, run_time_input, &results]() {
                results[i] = test_band_dtn_write(i + 1, buf_size, run_time_input);
            });
        }
        for (auto& th : threads) { th.join(); }
        long sum = 0;
        for (long res : results) { sum += res; }
        return sum;
    }
}

long test_band_nds_write_multi(ssize_t buf_size, int run_time_input, int thread_num)
{
    NdsFileError_t status = NdsFileDriverOpen();
    printf("Thread %d: device: %d, nds_driver_open ret(%d)\n", thread_num, device_id, status.err);
    if (thread_num <= 1) {
        long result = test_band_nds_write(0, buf_size, run_time_input);
        NdsFileDriverClose();
        return result;
    } else {
        std::vector<std::thread> threads;
        std::vector<long> results(thread_num);
        for (int i = 0; i < thread_num; ++i) {
            threads.emplace_back([i, buf_size, run_time_input, &results]() {
                results[i] = test_band_nds_write(i + 1, buf_size, run_time_input);
            });
        }
        for (auto& th : threads) { th.join(); }
        long sum = 0;
        for (long res : results) { sum += res; }
        NdsFileDriverClose();
        return sum;
    }
}

long test_band_dtn_read_multi(ssize_t buf_size, int run_time_input, int thread_num)
{
    if (thread_num <= 1) {
        return test_band_dtn_read(0, buf_size, run_time_input);
    } else {
        std::vector<std::thread> threads;
        std::vector<long> results(thread_num);
        for (int i = 0; i < thread_num; ++i) {
            threads.emplace_back([i, buf_size, run_time_input, &results]() {
                results[i] = test_band_dtn_read(i + 1, buf_size, run_time_input);
            });
        }
        for (auto& th : threads) { th.join(); }
        long sum = 0;
        for (long res : results) { sum += res; }
        return sum;
    }
}

long test_band_nds_read_multi(ssize_t buf_size, int run_time_input, int thread_num)
{
    NdsFileError_t status = NdsFileDriverOpen();
    printf("Thread %d: device: %d, nds_driver_open ret(%d)\n", thread_num, device_id, status.err);
    if (thread_num <= 1) {
        long result = test_band_nds_read(0, buf_size, run_time_input);
        NdsFileDriverClose();
        return result;
    } else {
        std::vector<std::thread> threads;
        std::vector<long> results(thread_num);
        for (int i = 0; i < thread_num; ++i) {
            threads.emplace_back([i, buf_size, run_time_input, &results]() {
                results[i] = test_band_nds_read(i + 1, buf_size, run_time_input);
            });
        }
        for (auto& th : threads) { th.join(); }
        long sum = 0;
        for (long res : results) { sum += res; }
        NdsFileDriverClose();
        return sum;
    }
}

int main(int argc, char* argv[])
{
    void* ptr = NULL;
    int runtime = 10;
    int threads = 1;
    long band_width = 0;
    std::string io_type_name;
    std::string xfer_type_name;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {  //-f 测试文件名
            file_name = argv[++i];
        } else if (arg == "-D" && i + 1 < argc) {  //-D 文件路径（f和D互斥）
            file_path = argv[++i];
        } else if (arg == "-d" && i + 1 < argc) {  //-d 设备id
            device_id = std::stoi(argv[++i]);
        } else if (arg == "-w" && i + 1 < argc) {  //-w 线程数 （可选）
            threads = std::stoi(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {  //-s 文件大小（支持K M G）
            file_size = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {  //-o 起始偏移量  （可选）
            start_offset = static_cast<off_t>(std::stoll(argv[++i]));
        } else if (arg == "-m" && i + 1 < argc) {  //-m NPU地址起始偏移量  （可选）
            ptr_offset = static_cast<off_t>(std::stoll(argv[++i]));
        } else if (arg == "-i" && i + 1 < argc) {  //-i IO大小 （支持K M G）
            io_size = argv[++i];
        } else if (arg == "-x" && i + 1 < argc) {  //-x 传输模式 （nds 0、cpugpu 1、cpu 2）
            xfer_type = std::stoi(argv[++i]);
        } else if (arg == "-I" &&
                   i + 1 < argc) {  //-I IO类型（顺序读 0 顺序写 1 随机读 2 随机写 3）
            transfer_type = std::stoi(argv[++i]);
        } else if (arg == "-T" && i + 1 < argc) {  //-T 持续时间 （s）
            runtime = std::stoi(argv[++i]);
        } else if (arg == "-b" && i + 1 < argc) {  // 申请io大小
            io_size_arl = argv[++i];
        } else if (arg == "-a" && i + 1 < argc) {  // 申请大页or普通页（0小页，1大页）
            application_page_size = std::stoi(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {  // 单步运行(0为多次，1为单次)
            number_runs = std::stoi(argv[++i]);
        }
    }
    if (runtime <= 0) {
        LOG_ERROR("Runtime(-T) cannot be negative.");
        return -1;
    }
    if (file_name.empty() && file_path.empty()) {
        LOG_ERROR("Please add file(-f) or folder(-D).");
        return -1;
    }

    file_byte_size = parse_size(file_size);
    buf_size = parse_size(io_size);
    if (io_size_arl.empty()) {
        buf_size_arl = buf_size;
    } else {
        buf_size_arl = parse_size(io_size_arl);
    }

    aclError ret = aclInit(nullptr);
    ret = aclrtSetDevice(device_id);
    if (ret) {
        LOG_ERROR("aclrtSetDevice failed, nds_driver initialization failed, ret:%d\n", ret);
        return -1;
    }

    if (xfer_type == 0) {
        xfer_type_name = "nds";
        if (transfer_type == 0 || transfer_type == 2) {
            band_width = test_band_nds_read_multi(buf_size, runtime, threads);
        } else if (transfer_type == 1 || transfer_type == 3) {
            band_width = test_band_nds_write_multi(buf_size, runtime, threads);
        } else {
            LOG_ERROR("transfer type(-I) error, [read 0, write 1, randread 2, randwrite 3]");
            return -1;
        }
    } else if (xfer_type == 1 || xfer_type == 2) {
        xfer_type_name = xfer_type == 1 ? "gpucpu" : "cpu";
        if (transfer_type == 0 || transfer_type == 2) {
            band_width = test_band_dtn_read_multi(buf_size, runtime, threads);
        } else if (transfer_type == 1 || transfer_type == 3) {
            band_width = test_band_dtn_write_multi(buf_size, runtime, threads);
        } else {
            LOG_ERROR("transfer type(-I) error, [read 0, write 1, randread 2, randwrite 3]");
            return -1;
        }
    } else {
        xfer_type_name = "unknown";
        LOG_ERROR("xfer type(-x) error, [nds 0, cpunpu 1, cpu 2]");
        return -1;
    }

    const char* io_type_names[] = {"read", "write", "randread", "randwrite"};
    if (transfer_type >= 0 && transfer_type <= 3) {
        io_type_name = io_type_names[transfer_type];
    } else {
        io_type_name = "unknown";
    }
    aclFinalize();
    printf(
        "********IoType: %s, XferType: %s, Threads: %d, IOSize: %s, Bandwitch:%ld M/s, TotalTime: "
        "%d s*********\n",
        io_type_name.c_str(), xfer_type_name.c_str(), threads, io_size.c_str(), band_width,
        runtime);
    return 0;
}
