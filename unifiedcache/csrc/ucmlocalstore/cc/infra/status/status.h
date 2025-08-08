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
#ifndef UCM_LOCAL_STORE_STATUS_H
#define UCM_LOCAL_STORE_STATUS_H

#include <cstdint>

namespace UCM {

enum class Status : int32_t {
#define UCM_LOCAL_STORE_MAKE_STATUS_CODE(i) (-50000 - (i))
    OK = 0,
    ERROR = -1,
    OSERROR = UCM_LOCAL_STORE_MAKE_STATUS_CODE(0),
    EXIST = UCM_LOCAL_STORE_MAKE_STATUS_CODE(1),
    NOENT = UCM_LOCAL_STORE_MAKE_STATUS_CODE(2),
    BUSY = UCM_LOCAL_STORE_MAKE_STATUS_CODE(3),
    RETRY = UCM_LOCAL_STORE_MAKE_STATUS_CODE(4),
    VERSION_UNMATCH = UCM_LOCAL_STORE_MAKE_STATUS_CODE(5),
    INVALID_PARAM = UCM_LOCAL_STORE_MAKE_STATUS_CODE(6),
    EMPTY = UCM_LOCAL_STORE_MAKE_STATUS_CODE(7),
    NOT_EXIST = UCM_LOCAL_STORE_MAKE_STATUS_CODE(8),
#undef UCM_LOCAL_STORE_MAKE_STATUS_CODE
};

inline int32_t Unwrap(Status s) { return static_cast<int32_t>(s); }

} // namespace UCM

#endif // UCM_LOCAL_STORE_STATUS_H

