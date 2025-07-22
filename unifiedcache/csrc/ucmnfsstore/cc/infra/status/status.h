/**
/* MIT License
/*
/* Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
/*
/* Permission is hereby granted, free of charge, to any person obtaining a copy
/* of this software and associated documentation files (the "Software"), to deal
/* in the Software without restriction, including without limitation the rights
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/* copies of the Software, and to permit persons to whom the Software is
/* furnished to do so, subject to the following conditions:
/*
/* The above copyright notice and this permission notice shall be included in all
/* copies or substantial portions of the Software.
/*
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/* SOFTWARE.
 * */
#ifndef UNIFIEDCACHE_STATUS_H
#define UNIFIEDCACHE_STATUS_H

#include <cstdint>

namespace UC {

class Status {
    enum class Code {
#define UC_MAKE_STATUS_CODE(i) (-50000 - (i))
        OK = 0,
        ERROR = -1,
        EPARAM = UC_MAKE_STATUS_CODE(0),
        EOOM = UC_MAKE_STATUS_CODE(1),
        EOSERROR = UC_MAKE_STATUS_CODE(2),
        EDUPLICATE = UC_MAKE_STATUS_CODE(3),
        ERETRY = UC_MAKE_STATUS_CODE(4),
        ENOOBJ = UC_MAKE_STATUS_CODE(5),
        ESERIALIZE = UC_MAKE_STATUS_CODE(6),
        EDESERIALIZE = UC_MAKE_STATUS_CODE(7),
        EUNSUPPORTED = UC_MAKE_STATUS_CODE(8),
#undef UC_MAKE_STATUS_CODE
    };

public:
    static Status& OK()
    {
        static Status s{Code::OK};
        return s;
    }
    static Status& Error()
    {
        static Status s{Code::ERROR};
        return s;
    }
    static Status& InvalidParam()
    {
        static Status s{Code::EPARAM};
        return s;
    }
    static Status& OutOfMemory()
    {
        static Status s{Code::EOOM};
        return s;
    }
    static Status& OsApiError()
    {
        static Status s{Code::EOSERROR};
        return s;
    }
    static Status& DuplicateKey()
    {
        static Status s{Code::EDUPLICATE};
        return s;
    }
    static Status& Retry()
    {
        static Status s{Code::ERETRY};
        return s;
    }
    static Status& NotFound()
    {
        static Status s{Code::ENOOBJ};
        return s;
    }
    static Status& SerializeFailed()
    {
        static Status s{Code::ESERIALIZE};
        return s;
    }
    static Status& DeserializeFailed()
    {
        static Status s{Code::EDESERIALIZE};
        return s;
    }
    static Status& Unsupported()
    {
        static Status s{Code::EUNSUPPORTED};
        return s;
    }

public:
    Status() : _code{Code::OK} {}
    Status(const Status& status) { this->_code = status._code; }
    Status& operator=(const Status& status)
    {
        if (this != &status) { this->_code = status._code; }
        return *this;
    }
    bool operator==(const Status& status) const { return this->_code == status._code; }
    bool operator!=(const Status& status) const { return this->_code != status._code; }
    int32_t Underlying() const { return static_cast<int32_t>(this->_code); }
    bool Success() const { return this->_code == Code::OK; }
    bool Failure() const { return this->_code != Code::OK; }

private:
    Status(const Code code) : _code{code} {}

private:
    Code _code;
};

inline int32_t format_as(const Status& status) { return status.Underlying(); }

} // namespace UC

#endif
