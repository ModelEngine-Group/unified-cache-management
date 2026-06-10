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
#ifndef TRANS_CASE_H
#define TRANS_CASE_H

#include <algorithm>
#include <cstdint>
#include <fmt/core.h>
#include <memory>
#include <string>
#include <vector>

enum class TransType : uint8_t { ANY, H2D, D2H };

struct TransCtx {
    std::size_t size;
    std::size_t num;
    std::size_t iter;
    std::size_t nDevice;
};

struct TransCase {
    std::string brief;
    TransType type;
    std::string host;
    std::string device;
    std::string method;

    TransCase(std::string brief, TransType type, std::string host, std::string device,
              std::string method)
        : brief{std::move(brief)},
          type{std::move(type)},
          host{std::move(host)},
          device{std::move(device)},
          method{std::move(method)}
    {
    }
    virtual ~TransCase() = default;
    virtual void Run(const TransCtx& ctx) const = 0;
};

class TransCaseFactory {
    TransCaseFactory() = default;
    std::vector<std::shared_ptr<TransCase>> cases_;

public:
    static TransCaseFactory& Instance()
    {
        static TransCaseFactory factory;
        return factory;
    }
    void Register(std::shared_ptr<TransCase> c) { cases_.push_back(std::move(c)); }
    void ShowAllCases() const
    {
        fmt::println("[[ ALL CASES ]]");
        auto formatPrefix = [](const auto& c) {
            auto typeStr = c->type == TransType::H2D ? "[H2D]" : "[D2H]";
            auto src = c->type == TransType::H2D ? c->host : c->device;
            auto dst = c->type == TransType::H2D ? c->device : c->host;
            return fmt::format("{} {} --({})--> {}", typeStr, src, c->method, dst);
        };
        auto sorted = cases_;
        std::stable_sort(sorted.begin(), sorted.end(),
                         [](const auto& a, const auto& b) { return a->type < b->type; });
        size_t maxLen = 0;
        for (const auto& c : sorted) { maxLen = std::max(maxLen, formatPrefix(c).length()); }
        for (const auto& c : sorted) {
            auto prefix = formatPrefix(c);
            fmt::println("  {}{} : {}", prefix, std::string(maxLen - prefix.length(), ' '),
                         c->brief);
        }
    }
    std::vector<std::shared_ptr<TransCase>> Filter(const TransType& type,
                                                   const std::vector<std::string>& hosts,
                                                   const std::vector<std::string>& devices,
                                                   const std::vector<std::string>& methods) const
    {
        std::vector<std::shared_ptr<TransCase>> result;
        auto contains = [](const auto& list, const auto& val) {
            return list.empty() || std::find(list.begin(), list.end(), val) != list.end();
        };
        for (const auto& c : cases_) {
            if (type != TransType::ANY && c->type != type) continue;
            if (!contains(hosts, c->host)) continue;
            if (!contains(devices, c->device)) continue;
            if (!contains(methods, c->method)) continue;
            result.push_back(c);
        }
        return result;
    }
};

template <typename T>
class Registrar {
    static_assert(std::is_base_of_v<TransCase, T>, "T must inherit TransCase");

public:
    Registrar() { TransCaseFactory::Instance().Register(std::make_shared<T>()); }
};

#define DEFINE_TRANS_CASE(ClassName, Brief, Type, Host, Device, Method, Ctx) \
    class ClassName : public TransCase {                                     \
    public:                                                                  \
        ClassName() : TransCase(Brief, Type, Host, Device, Method) {}        \
        void Run(const TransCtx&) const override;                            \
    };                                                                       \
    static Registrar<ClassName> global_##ClassName##_registrar;              \
    void ClassName::Run(const TransCtx& Ctx) const

#define DEFINE_TRANS_H2D_CASE(Brief, Host, Device, Method, Ctx)                                   \
    DEFINE_TRANS_CASE(_H2D_##Host##_To_##Device##_With_##Method##_, Brief, TransType::H2D, #Host, \
                      #Device, #Method, Ctx)

#define DEFINE_TRANS_D2H_CASE(Brief, Device, Host, Method, Ctx)                                   \
    DEFINE_TRANS_CASE(_D2H_##Device##_To_##Host##_With_##Method##_, Brief, TransType::D2H, #Host, \
                      #Device, #Method, Ctx)

#endif
