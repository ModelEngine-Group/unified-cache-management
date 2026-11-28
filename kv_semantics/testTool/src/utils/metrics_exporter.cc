/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
 * SOFTWARE. */

#include "utils/metrics_exporter.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <fmt/format.h>

namespace {

/** 指标接口名（下标与 worker/executor.h OpType 枚举值对齐） */
const char *const kMetricsOpNames[kMetricsOpCount] = {"exist", "put", "get", "delete"};

/** summary 输出的分位点序列（与 HELP 说明保持一致） */
constexpr double kSummaryQuantiles[] = {0.5, 0.9, 0.99};

/**
 * 组装 HTTP/1.1 响应报文：状态行 + 头 + 正文
 * @param status 状态描述（如 "200 OK" / "500 Internal Server Error"）
 * @param contentType Content-Type 头值
 * @param body 响应正文
 */
std::string BuildHttpResponse(const std::string &status, const std::string &contentType,
                              const std::string &body)
{
    return fmt::format(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n"
        "Connection: close\r\n\r\n{}",
        status, contentType, body.size(), body);
}

#if !defined(_WIN32)
/**
 * 循环发送直到完成或失败（处理 partial send）
 * @return true 全部发送成功
 */
bool SendAll(int fd, const char *data, std::size_t size)
{
    std::size_t sent = 0;
    while (sent < size) {
        const ssize_t n = ::send(fd, data + sent, size - sent, MSG_NOSIGNAL);
        if (n < 0) { return false; }
        sent += static_cast<std::size_t>(n);
    }
    return true;
}
#endif

}  // namespace

std::uint64_t LatencyQuantileUs(std::vector<std::uint32_t> &samples, double quantile)
{
    if (samples.empty()) { return 0; }
    std::sort(samples.begin(), samples.end());
    const double rank = std::ceil(quantile * static_cast<double>(samples.size()));
    std::size_t index = static_cast<std::size_t>(rank);
    if (index > samples.size()) { index = samples.size(); }
    return samples[index - 1];
}

std::string RenderPrometheusText(const MetricsSnapshot &snapshot)
{
    std::string text;
    text.reserve(4096);
    // ---- summary：分接口时延（分位 / 累计和 / 样本数）----
    text +=
        "# HELP testtool_io_duration_seconds Task-level IO latency (submit to completion) by "
        "operation.\n";
    text += "# TYPE testtool_io_duration_seconds summary\n";
    for (std::size_t op = 0; op < kMetricsOpCount; ++op) {
        const char *const name = kMetricsOpNames[op];
        // 样本副本排序（快照入参只读），滚动最近样本的分位近似
        std::vector<std::uint32_t> samples = snapshot.latencySamplesUs[op];
        for (const double q : kSummaryQuantiles) {
            const std::uint64_t us = LatencyQuantileUs(samples, q);
            if (snapshot.tasksByOp[op] == 0) { continue; }  // 无样本接口不输出分位行
            text += fmt::format(
                "testtool_io_duration_seconds{{op=\"{}\",quantile=\"{}\"}} "
                "{:.9}\n",
                name, q, static_cast<double>(us) / 1e6);
        }
        text += fmt::format("testtool_io_duration_seconds_sum{{op=\"{}\"}} {:.9}\n", name,
                            static_cast<double>(snapshot.latencyNsByOp[op]) / 1e9);
        text += fmt::format("testtool_io_duration_seconds_count{{op=\"{}\"}} {}\n", name,
                            snapshot.tasksByOp[op]);
    }
    // ---- counter 族：分接口操作数与流量 ----
    text +=
        "# HELP testtool_ops_total Key-level operations by operation type (including "
        "not-found entries).\n";
    text += "# TYPE testtool_ops_total counter\n";
    for (std::size_t op = 0; op < kMetricsOpCount; ++op) {
        text += fmt::format("testtool_ops_total{{op=\"{}\"}} {}\n", kMetricsOpNames[op],
                            snapshot.opsByOp[op]);
    }
    text +=
        "# HELP testtool_io_bytes_total Value payload bytes by operation type (successful "
        "entries).\n";
    text += "# TYPE testtool_io_bytes_total counter\n";
    for (std::size_t op = 0; op < kMetricsOpCount; ++op) {
        text += fmt::format("testtool_io_bytes_total{{op=\"{}\"}} {}\n", kMetricsOpNames[op],
                            snapshot.bytesByOp[op]);
    }
    // ---- counter / gauge：全局观察计数与运行时长 ----
    text +=
        "# HELP testtool_errors_total Failed operations (submit/wait failures and entry "
        "errors, not-found observations excluded).\n";
    text += "# TYPE testtool_errors_total counter\n";
    text += fmt::format("testtool_errors_total {}\n", snapshot.errorsTotal);
    text +=
        "# HELP testtool_notfound_total NOT_FOUND observations (get/delete on absent "
        "keys).\n";
    text += "# TYPE testtool_notfound_total counter\n";
    text += fmt::format("testtool_notfound_total {}\n", snapshot.notFoundTotal);
    text += "# HELP testtool_uptime_seconds Seconds since the scheduler run started.\n";
    text += "# TYPE testtool_uptime_seconds gauge\n";
    text += fmt::format("testtool_uptime_seconds {:.3}\n", snapshot.uptimeSecs);
    return text;
}

MetricsExporter::~MetricsExporter() { Stop(); }

bool MetricsExporter::Start(std::function<std::string()> render, int port, std::string &err)
{
#if defined(_WIN32)
    (void)render;
    (void)port;
    err = "metrics exporter unsupported on non-POSIX platform";
    return false;
#else
    if (running_.load(std::memory_order_relaxed)) {
        err = "metrics exporter already running";
        return false;
    }
    render_ = std::move(render);
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        err = std::string("metrics exporter socket failed: ") + std::strerror(errno);
        return false;
    }
    int reuse = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<std::uint16_t>(port));
    if (::bind(fd, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) != 0 ||
        ::listen(fd, 8) != 0) {
        err = fmt::format("metrics exporter listen on port {} failed: {}", port,
                          std::strerror(errno));
        ::close(fd);
        return false;
    }
    listenFd_ = fd;
    running_.store(true, std::memory_order_relaxed);
    acceptThread_ = std::thread(&MetricsExporter::AcceptLoop, this);
    return true;
#endif
}

void MetricsExporter::Stop()
{
    if (!running_.exchange(false, std::memory_order_relaxed)) { return; }
#if !defined(_WIN32)
    // shutdown 触发阻塞中的 accept 返回，循环见退出标志结束
    if (listenFd_ >= 0) {
        ::shutdown(listenFd_, 2 /* SHUT_RDWR */);
        ::close(listenFd_);
        listenFd_ = -1;
    }
#endif
    if (acceptThread_.joinable()) { acceptThread_.join(); }
}

#if !defined(_WIN32)
void MetricsExporter::AcceptLoop()
{
    while (running_.load(std::memory_order_relaxed)) {
        sockaddr_in peer{};
        socklen_t peerLen = sizeof(peer);
        const int conn = ::accept(listenFd_, reinterpret_cast<sockaddr *>(&peer), &peerLen);
        if (conn < 0) {
            if (!running_.load(std::memory_order_relaxed)) { return; }  // Stop 触发的返回
            continue;                                                   // 瞬时错误重试
        }
        // 读一段请求头即够（不解析路径，一律返回指标文本）
        char discard[2048];
        (void)::recv(conn, discard, sizeof(discard), 0);
        std::string body;
        std::string status = "200 OK";
        try {
            body = render_ ? render_() : std::string();
        } catch (const std::exception &e) {
            body = std::string("# render failed: ") + e.what();
            status = "500 Internal Server Error";
        }
        const std::string response =
            BuildHttpResponse(status, "text/plain; version=0.0.4; charset=utf-8", body);
        SendAll(conn, response.data(), response.size());
        ::close(conn);
    }
}
#endif
