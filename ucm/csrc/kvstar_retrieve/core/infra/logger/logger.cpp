#include <spdlog/cfg/helpers.h>
#include <spdlog/details/os.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "logger.h"

namespace KVStar {

static std::mutex g_mutex;
static std::shared_ptr<spdlog::logger> g_logger = nullptr;

std::shared_ptr<spdlog::logger> Logger::Make()
{
    if (g_logger) { return g_logger; }
    std::unique_lock lock(g_mutex);
    if (g_logger) { return g_logger; }
    try {
        const std::string name = "KVSTAR_RETRIEVE";
        const std::string envLevel = name + "_LOGGER_LEVEL";
        g_logger = spdlog::stdout_color_mt(name);
        g_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f %z] [%n] [%^%L%$] %v [PID: %P, TID: %t] [%s:%#,%!]");
        auto level = spdlog::details::os::getenv(envLevel.c_str());
        if (!level.empty()) { spdlog::cfg::helpers::load_levels(level); }
        return g_logger;
    } catch (...) {
        return spdlog::default_logger();
    }
}

}