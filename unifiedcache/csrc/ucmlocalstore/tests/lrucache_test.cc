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
#include "lrucache/lrucache.h"
#include <thread>
#include <random>
#include <vector>
#include <unordered_set>
#include <gtest/gtest.h>

class UCMLocalStoreLRUCacheTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        system("rm -f /dev/shm/*");
        std::thread t1([] {
            EXPECT_EQ(UCM::Status::OK, lru.Initialize(cache_num, cache_size));
        });
        std::thread t2([] {
            EXPECT_EQ(UCM::Status::OK, lru.Initialize(cache_num, cache_size));
        });
        std::thread t3([] {
            EXPECT_EQ(UCM::Status::OK, lru.Initialize(cache_num, cache_size));
        });
        std::thread t4([] {
            while (cache_ids.size() != 2730) {
                const std::string charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<size_t> dis(0, charset.size() - 1);
                
                std::string cache_id;
                for (size_t i = 0; i < 16; ++i) {
                    cache_id += charset[dis(gen)];
                }
                cache_ids.insert(cache_id);
            }
        });
        t1.join();
        t2.join();
        t3.join();
        t4.join();
    }

    static void TearDownTestSuite()
    {
        system("rm -f /dev/shm/*");
    }

    static inline uint32_t cache_num{8192};
    static inline uint32_t cache_size{1 << 20};
    static inline UCM::LRUCache lru;
    static inline std::unordered_set<std::string> cache_ids;
};

TEST_F(UCMLocalStoreLRUCacheTest, InsertAndDone)
{
    std::thread t1([] {
        for (std::string_view cache_id : cache_ids) {
            void* cache_data = nullptr;
            auto s = lru.Insert(cache_id, cache_data);
            ASSERT_EQ(s == UCM::Status::OK || s == UCM::Status::EXIST, true);
            if (s == UCM::Status::OK) {
                lru.Done(cache_data);
            }
        }
    });
    std::thread t2([] {
        for (std::string_view cache_id : cache_ids) {
            void* cache_data = nullptr;
            auto s = lru.Insert(cache_id, cache_data);
            ASSERT_EQ(s == UCM::Status::OK || s == UCM::Status::EXIST, true);
            if (s == UCM::Status::OK) {
                lru.Done(cache_data);
            }
        }
    });
    std::thread t3([] {
        for (std::string_view cache_id : cache_ids) {
            void* cache_data = nullptr;
            auto s = lru.Insert(cache_id, cache_data);
            ASSERT_EQ(s == UCM::Status::OK || s == UCM::Status::EXIST, true);
            if (s == UCM::Status::OK) {
                lru.Done(cache_data);
            }
        }
    });
    t1.join();
    t2.join();
    t3.join();
}

TEST_F(UCMLocalStoreLRUCacheTest, FindAndDone)
{
    std::thread t1([]() {
        for (std::string_view cache_id : cache_ids) {
            void* cache_data = nullptr;
            ASSERT_EQ(UCM::Status::OK, lru.Find(cache_id, cache_data));
            lru.Done(cache_data);
        }
    });
    std::thread t2([]() {
        for (std::string_view cache_id : cache_ids) {
            void* cache_data = nullptr;
            ASSERT_EQ(UCM::Status::OK, lru.Find(cache_id, cache_data));
            lru.Done(cache_data);
        }
    });
    std::thread t3([]() {
        for (std::string_view cache_id : cache_ids) {
            void* cache_data = nullptr;
            ASSERT_EQ(UCM::Status::OK, lru.Find(cache_id, cache_data));
            lru.Done(cache_data);
        }
    });
    t1.join();
    t2.join();
    t3.join();
}