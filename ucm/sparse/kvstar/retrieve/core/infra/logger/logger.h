#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_LOGGER_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_LOGGER_H

#include <spdlog/spdlog.h>

namespace KVStar {

class Logger {
public:
    static std::shared_ptr<spdlog::logger> Make();
};

}

#define KVSTAR_LOG(level, ...)                                                                                             \
KVStar::Logger::Make()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, level, __VA_ARGS__)
#define KVSTAR_DEBUG(...) KVSTAR_LOG(spdlog::level::debug, __VA_ARGS__)
#define KVSTAR_INFO(...) KVSTAR_LOG(spdlog::level::info, __VA_ARGS__)
#define KVSTAR_WARN(...) KVSTAR_LOG(spdlog::level::warn, __VA_ARGS__)
#define KVSTAR_ERROR(...) KVSTAR_LOG(spdlog::level::err, __VA_ARGS__)


#endif //UCM_SPARSE_KVSTAR_RETRIEVE_LOGGER_H