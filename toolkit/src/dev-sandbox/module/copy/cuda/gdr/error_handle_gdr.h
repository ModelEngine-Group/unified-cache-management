/**
 * MIT License
 *
 * Copyright (c) 2026 relat-ivity
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
#ifndef ERROR_HANDLE_GDR_H
#define ERROR_HANDLE_GDR_H

#include <infiniband/verbs.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "../error_handle_cuda.h"

#define GDR_IBV_ASSERT(expr)                                                              \
    do {                                                                                  \
        const int __ibvRc = (expr);                                                       \
        if (__ibvRc != 0) {                                                               \
            std::stringstream errmsg;                                                     \
            errmsg << "[" << __ibvRc << "] " << std::strerror(__ibvRc)                   \
                   << " in expression " << #expr << " at " << __PRETTY_FUNCTION__        \
                   << "() : " << __FILE__ << ":" << __LINE__ << std::endl;               \
            fprintf(stderr, "%s", errmsg.str().c_str());                                 \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

#define GDR_WC_ASSERT(status, wrId)                                                       \
    do {                                                                                  \
        if ((status) != IBV_WC_SUCCESS) {                                                 \
            std::stringstream errmsg;                                                     \
            errmsg << "[WC " << (wrId) << "] " << ibv_wc_status_str(status)              \
                   << " at " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":"        \
                   << __LINE__ << std::endl;                                              \
            fprintf(stderr, "%s", errmsg.str().c_str());                                 \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

#endif  // ERROR_HANDLE_GDR_H
