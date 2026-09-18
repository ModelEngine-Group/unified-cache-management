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
#ifndef TRANS_ASSERT_ASCEND_H
#define TRANS_ASSERT_ASCEND_H

#include <acl/acl.h>
#include <sstream>

#define TRANS_ASSERT(expr)                                                                  \
    do {                                                                                    \
        if (!(expr)) {                                                                      \
            fprintf(stderr, "Assertion failed: %s, at %s:%d\n", #expr, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

#define ASCEND_ASSERT(expr)                                                           \
    do {                                                                              \
        auto __aclErr = (expr);                                                       \
        if ((__aclErr) != ACL_SUCCESS) {                                              \
            std::stringstream errmsg;                                                 \
            errmsg << "[ACL Error " << __aclErr << "] "                               \
                   << (aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "Unknown error") \
                   << " in expression " << #expr << " at " << __PRETTY_FUNCTION__     \
                   << "() : " << __FILE__ << ":" << __LINE__ << std::endl;            \
            fprintf(stderr, "%s", errmsg.str().c_str());                              \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

#endif  // TRANS_ASSERT_ASCEND_H
