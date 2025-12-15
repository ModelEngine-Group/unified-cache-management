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
#ifndef UNIFIEDCACHE_TEST_DETAIL_PATH_BASE_H
#define UNIFIEDCACHE_TEST_DETAIL_PATH_BASE_H

#include <gtest/gtest.h>
#include "random.h"

namespace UC::Test::Detail {

class PathBase : public ::testing::Test {
public:
    void SetUp() override
    {
        testing::Test::SetUp();
        const auto info = testing::UnitTest::GetInstance()->current_test_info();
        std::string testCaseName = info->test_case_name();
        std::string testName = info->name();
        this->path_ = "./" + testCaseName + "_" + testName + "_" + this->rd_.RandomString(20) + "/";
        system((std::string("rm -rf ") + this->path_).c_str());
        system((std::string("mkdir -p ") + this->path_).c_str());
    }
    void TearDown() override
    {
        system((std::string("rm -rf ") + this->path_).c_str());
        testing::Test::TearDown();
    }
    std::string Path() const { return this->path_; }

private:
    Random rd_;
    std::string path_;
};

}  // namespace UC::Test::Detail

#endif
