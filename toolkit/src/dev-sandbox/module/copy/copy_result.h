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
#ifndef COPY_RESULT_H
#define COPY_RESULT_H

#include <algorithm>
#include <fmt/format.h>
#include <numeric>
#include <string>
#include <vector>

class CopyResult {
public:
    struct Result {
        std::string src;
        std::string dst;
        std::string method;
        size_t size;
        size_t count;
        struct {
            size_t min;
            size_t max;
            size_t avg;
            size_t p50;
            size_t p90;

            void Parse(std::vector<size_t>&& costs)
            {
                if (costs.empty()) return;
                size_t sum = 0;
                min = max = costs[0];
                for (auto cost : costs) {
                    if (cost < min) min = cost;
                    if (cost > max) max = cost;
                    sum += cost;
                }
                avg = sum / costs.size();
                std::sort(costs.begin(), costs.end());
                p50 = costs[costs.size() / 2];
                p90 = costs[costs.size() * 9 / 10];
            }
            std::string ToString() const
            {
                return fmt::format("{} / {} / {} / {} / {}", min, max, avg, p50, p90);
            }
        } submit, copy;
        Result(std::string src, std::string dst, std::string method, size_t size, size_t count,
               std::vector<size_t>&& submitCosts, std::vector<size_t>&& copyCosts)
            : src(std::move(src)),
              dst(std::move(dst)),
              method(std::move(method)),
              size(size),
              count(count)
        {
            submit.Parse(std::move(submitCosts));
            copy.Parse(std::move(copyCosts));
        }
    };
    void Push(Result&& result) { results_.push_back(std::move(result)); }
    void Show(std::string title) const
    {
        const std::string indentation = "  ";
        fmt::println(title);
        fmt::println("{}{:<18}{:<18}{:<10}{:<10}{:<8}{:<40}{:<44}{}", indentation, "From", "To",
                     "Method", "Size(KB)", "Count", "Submit(us)-(Min/Max/Avg/P50/P90)",
                     "Copy(us)-(Min/Max/Avg/P50/P90)", "BW(GB/s)");
        for (const auto& result : results_) {
            auto bw =
                result.size * result.count * 1e6f / result.copy.avg / 1024.f / 1024.f / 1024.f;
            fmt::println("{}{:<18}{:<18}{:<10}{:<10.0f}{:<8}{:<40}{:<44}{:.3f}", indentation,
                         result.src, result.dst, result.method, result.size / 1024.f, result.count,
                         result.submit.ToString(), result.copy.ToString(), bw);
        }
    }

private:
    std::vector<Result> results_;
};

#endif  // COPY_RESULT_H
