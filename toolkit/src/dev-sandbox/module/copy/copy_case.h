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
#ifndef COPY_CASE_H
#define COPY_CASE_H

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

class CopyCase {
    std::string key_;
    std::string brief_;

public:
    struct Context {
        size_t size;
        size_t num;
        size_t iter;
        size_t nDevice;
    };
    CopyCase(std::string key, std::string brief) : key_{std::move(key)}, brief_{std::move(brief)} {}
    virtual ~CopyCase() = default;
    virtual void Run(const Context& ctx) const = 0;
    const std::string& Key() const { return key_; }
    const std::string& Brief() const { return brief_; }
};

class CopyCaseFactory {
    CopyCaseFactory() = default;
    std::vector<std::shared_ptr<CopyCase>> cases_;

public:
    static CopyCaseFactory& Instance()
    {
        static CopyCaseFactory factory;
        return factory;
    }
    void Register(std::shared_ptr<CopyCase> c) { cases_.push_back(std::move(c)); }
    const std::vector<std::shared_ptr<CopyCase>>& AllCases() const { return cases_; }
    std::vector<std::shared_ptr<CopyCase>> Filter(
        const std::unordered_set<std::string>& names) const
    {
        std::vector<std::shared_ptr<CopyCase>> result;
        if (names.empty()) { return cases_; }
        for (auto& c : cases_) {
            if (names.find(c->Key()) != names.end()) { result.push_back(c); }
        }
        return result;
    }
};

template <typename T>
class Registrar {
    static_assert(std::is_base_of_v<CopyCase, T>, "T must inherit CopyCase");

public:
    Registrar() { CopyCaseFactory::Instance().Register(std::make_shared<T>()); }
};

#define DEFINE_COPY_CASE(ClassName, Key, Brief, Ctx)            \
    class ClassName : public CopyCase {                         \
    public:                                                     \
        ClassName() : CopyCase(Key, Brief) {}                   \
        void Run(const Context&) const override;                \
    };                                                          \
    static Registrar<ClassName> global_##ClassName##_registrar; \
    void ClassName::Run(const Context& Ctx) const

#endif
