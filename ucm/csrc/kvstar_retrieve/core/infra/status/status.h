#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_STATUS_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_STATUS_H

#include <cstdint>

namespace KVStar {

class Status {
    enum class Code {
#define KVSTAR_MAKE_STATUS_CODE(i) (-50000 - (i))
        OK = 0,
        ERROR = -1,
        EPARAM = KVSTAR_MAKE_STATUS_CODE(0),
        EOOM = KVSTAR_MAKE_STATUS_CODE(1),
        EOSERROR = KVSTAR_MAKE_STATUS_CODE(2),
        ENOOBJ = KVSTAR_MAKE_STATUS_CODE(3),
        EUNSUPPORTED = KVSTAR_MAKE_STATUS_CODE(4),
#undef KVSTAR_MAKE_STATUS_CODE
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
    static Status& NotFound()
    {
        static Status s{Code::ENOOBJ};
        return s;
    }
    static Status& Unsupported()
    {
        static Status s{Code::EUNSUPPORTED};
        return s;
    }

public:
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

}


#endif //UCM_SPARSE_KVSTAR_RETRIEVE_STATUS_H