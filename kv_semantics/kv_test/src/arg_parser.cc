#include "arg_parser.h"
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace kv::bench {

std::string CommandTypeName(CommandType command)
{
    switch (command) {
        case CommandType::CONNECT: return "connect";
        case CommandType::CONFIG_CHECK: return "config check";
        case CommandType::STORE: return "store";
        case CommandType::RETRIEVE: return "retrieve";
        case CommandType::DELETE: return "delete";
        case CommandType::EXIST: return "exist";
        case CommandType::BATCH_STORE: return "batch-store";
        case CommandType::BATCH_RETRIEVE: return "batch-retrieve";
        case CommandType::POWER_CYCLE_PREPARE: return "power-cycle prepare";
        case CommandType::POWER_CYCLE_VERIFY: return "power-cycle verify";
        case CommandType::BENCH: return "bench";
        case CommandType::VERSION: return "version";
        case CommandType::UNKNOWN:
        default: return "unknown";
    }
}

std::string BenchOpTypeName(BenchOpType op)
{
    switch (op) {
        case BenchOpType::STORE: return "store";
        case BenchOpType::RETRIEVE: return "retrieve";
        case BenchOpType::BATCH_STORE: return "batch-store";
        case BenchOpType::BATCH_RETRIEVE: return "batch-retrieve";
        case BenchOpType::MIX: return "mix";
        case BenchOpType::UNKNOWN:
        default: return "unknown";
    }
}

namespace {

constexpr int kExitSuccess = 0;
constexpr int kExitInvalidArgument = 1;

bool IsOption(const std::string& argument)
{
    return argument.rfind("--", 0) == 0 || argument == "-h";
}

std::vector<std::string> SplitCommaList(const std::string& value)
{
    std::vector<std::string> items;
    std::string item;
    std::stringstream stream(value);
    while (std::getline(stream, item, ',')) { items.push_back(item); }
    return items;
}

Status ParseUint64(const std::string& option, const std::string& value, std::uint64_t& output)
{
    if (value.empty() || value[0] == '-') {
        return Status::Error(kExitInvalidArgument, option + " expects a non-negative integer");
    }

    char* end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno == ERANGE || end == value.c_str() || *end != '\0') {
        return Status::Error(kExitInvalidArgument, option + " expects a non-negative integer");
    }

    output = static_cast<std::uint64_t>(parsed);
    return Status::Success();
}

Status ParseUint32(const std::string& option, const std::string& value, std::uint32_t& output)
{
    std::uint64_t parsed = 0;
    auto status = ParseUint64(option, value, parsed);
    if (!status.Ok()) { return status; }
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        return Status::Error(kExitInvalidArgument, option + " is too large");
    }

    output = static_cast<std::uint32_t>(parsed);
    return Status::Success();
}

CommandType ParseCommand(const std::string& command)
{
    static const std::unordered_map<std::string, CommandType> kCommands = {
        {"connect",        CommandType::CONNECT       },
        {"version",        CommandType::VERSION       },
        {"store",          CommandType::STORE         },
        {"retrieve",       CommandType::RETRIEVE      },
        {"delete",         CommandType::DELETE        },
        {"exist",          CommandType::EXIST         },
        {"batch-store",    CommandType::BATCH_STORE   },
        {"batch-retrieve", CommandType::BATCH_RETRIEVE},
        {"bench",          CommandType::BENCH         },
    };

    auto iter = kCommands.find(command);
    return iter == kCommands.end() ? CommandType::UNKNOWN : iter->second;
}

BenchOpType ParseBenchOp(const std::string& operation)
{
    static const std::unordered_map<std::string, BenchOpType> kBenchOps = {
        {"store",          BenchOpType::STORE         },
        {"retrieve",       BenchOpType::RETRIEVE      },
        {"batch-store",    BenchOpType::BATCH_STORE   },
        {"batch-retrieve", BenchOpType::BATCH_RETRIEVE},
        {"mix",            BenchOpType::MIX           },
    };

    auto iter = kBenchOps.find(operation);
    return iter == kBenchOps.end() ? BenchOpType::UNKNOWN : iter->second;
}

Status SetCommand(const std::vector<std::string>& positionals, CommandOptions& options)
{
    if (positionals.empty()) {
        return Status::Error(kExitInvalidArgument, "missing kv-test command");
    }

    if (positionals[0] == "config") {
        if (positionals.size() != 2) {
            return Status::Error(kExitInvalidArgument, "config expects check subcommand");
        }
        if (positionals[1] == "check") {
            options.command = CommandType::CONFIG_CHECK;
            return Status::Success();
        }
        return Status::Error(kExitInvalidArgument, "unknown config subcommand: " + positionals[1]);
    }

    if (positionals[0] == "power-cycle") {
        if (positionals.size() != 2) {
            return Status::Error(kExitInvalidArgument,
                                 "power-cycle expects prepare or verify subcommand");
        }
        if (positionals[1] == "prepare") {
            options.command = CommandType::POWER_CYCLE_PREPARE;
            return Status::Success();
        }
        if (positionals[1] == "verify") {
            options.command = CommandType::POWER_CYCLE_VERIFY;
            return Status::Success();
        }
        return Status::Error(kExitInvalidArgument,
                             "unknown power-cycle subcommand: " + positionals[1]);
    }

    options.command = ParseCommand(positionals[0]);
    if (options.command == CommandType::UNKNOWN) {
        return Status::Error(kExitInvalidArgument, "unknown kv-test command: " + positionals[0]);
    }

    if (options.command == CommandType::BENCH && positionals.size() == 2) {
        const auto positionalBenchOp = ParseBenchOp(positionals[1]);
        if (positionalBenchOp == BenchOpType::UNKNOWN) {
            return Status::Error(kExitInvalidArgument,
                                 "unknown bench operation: " + positionals[1]);
        }
        if (options.benchOp != BenchOpType::UNKNOWN && options.benchOp != positionalBenchOp) {
            return Status::Error(kExitInvalidArgument, "bench operation conflicts with --bench-op");
        }
        options.benchOp = positionalBenchOp;
        return Status::Success();
    }

    if (positionals.size() != 1) {
        return Status::Error(kExitInvalidArgument,
                             "unexpected positional argument: " + positionals[1]);
    }

    return Status::Success();
}

}  // namespace

Status ArgParser::Parse(int argc, char** argv, CommandOptions& options) const
{
    options = CommandOptions{};
    std::vector<std::string> positionals;
    bool hasKey = false;
    bool hasKeys = false;
    bool hasKeysFile = false;
    bool hasCount = false;

    for (int index = 1; index < argc; ++index) {
        std::string argument = argv[index];
        if (!IsOption(argument)) {
            positionals.push_back(argument);
            continue;
        }

        std::string option = argument;
        std::string value;
        bool hasInlineValue = false;
        const auto equalPos = argument.find('=');
        if (equalPos != std::string::npos) {
            option = argument.substr(0, equalPos);
            value = argument.substr(equalPos + 1);
            hasInlineValue = true;
        }

        auto requireValue = [&]() -> Status {
            if (hasInlineValue) {
                if (value.empty()) {
                    return Status::Error(kExitInvalidArgument, option + " expects a value");
                }
                return Status::Success();
            }
            if (index + 1 >= argc || IsOption(argv[index + 1])) {
                return Status::Error(kExitInvalidArgument, option + " expects a value");
            }
            value = argv[++index];
            return Status::Success();
        };

        if (option == "--help" || option == "-h") {
            if (hasInlineValue) {
                return Status::Error(kExitInvalidArgument, option + " does not take a value");
            }
            options.helpRequested = true;
        } else if (option == "--version") {
            if (hasInlineValue) {
                return Status::Error(kExitInvalidArgument, "--version does not take a value");
            }
            options.versionRequested = true;
            options.command = CommandType::VERSION;
        } else if (option == "--check") {
            if (hasInlineValue) {
                return Status::Error(kExitInvalidArgument, "--check does not take a value");
            }
            options.check = true;
        } else if (option == "--progress") {
            if (hasInlineValue) {
                return Status::Error(kExitInvalidArgument, "--progress does not take a value");
            }
            options.progress = true;
        } else if (option == "--configpath") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            options.configPath = value;
        } else if (option == "--key") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            hasKey = true;
            options.singleKeyRequested = true;
            options.keys.push_back(value);
        } else if (option == "--keys") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            hasKeys = true;
            auto keys = SplitCommaList(value);
            for (const auto& key : keys) {
                if (key.empty()) {
                    return Status::Error(kExitInvalidArgument, "--keys contains an empty key");
                }
                options.keys.push_back(key);
            }
        } else if (option == "--keys-file") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            hasKeysFile = true;
            options.keysFile = value;
        } else if (option == "--prefix") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            options.keyPrefix = value;
        } else if (option == "--key-start") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            options.keyStartSet = true;
            status = ParseUint64(option, value, options.keyStart);
            if (!status.Ok()) { return status; }
        } else if (option == "--key-end") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            options.keyEndSet = true;
            status = ParseUint64(option, value, options.keyEnd);
            if (!status.Ok()) { return status; }
        } else if (option == "--count") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            hasCount = true;
            status = ParseUint64(option, value, options.count);
            if (!status.Ok()) { return status; }
        } else if (option == "--seed") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            status = ParseUint64(option, value, options.seed);
            if (!status.Ok()) { return status; }
        } else if (option == "--value-size" || option == "--io-size") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            status = ParseUint64(option, value, options.valueSize);
            if (!status.Ok()) { return status; }
        } else if (option == "--batch-size") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            status = ParseUint32(option, value, options.batchSize);
            if (!status.Ok()) { return status; }
        } else if (option == "--timeout") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            status = ParseUint64(option, value, options.timeoutMs);
            if (!status.Ok()) { return status; }
        } else if (option == "--output") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            options.outputPath = value;
        } else if (option == "--bench-op" || option == "--op") {
            // TODO(#12): Align bench CLI parameter names with the final kv-test CLI spec.
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            options.benchOp = ParseBenchOp(value);
            if (options.benchOp == BenchOpType::UNKNOWN) {
                return Status::Error(kExitInvalidArgument, "unknown bench operation: " + value);
            }
        } else if (option == "--concurrency") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            status = ParseUint32(option, value, options.concurrency);
            if (!status.Ok()) { return status; }
        } else if (option == "--duration") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            status = ParseUint64(option, value, options.durationSec);
            if (!status.Ok()) { return status; }
        } else if (option == "--warmup") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            status = ParseUint64(option, value, options.warmupSec);
            if (!status.Ok()) { return status; }
        } else if (option == "--read-ratio") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            status = ParseUint32(option, value, options.readRatio);
            if (!status.Ok()) { return status; }
        } else if (option == "--write-ratio") {
            auto status = requireValue();
            if (!status.Ok()) { return status; }
            status = ParseUint32(option, value, options.writeRatio);
            if (!status.Ok()) { return status; }
        } else {
            return Status::Error(kExitInvalidArgument, "unknown option: " + option);
        }
    }

    if (options.versionRequested) {
        if (!positionals.empty()) {
            return Status::Error(kExitInvalidArgument,
                                 "--version does not take positional arguments");
        }
        return Status::Success();
    }
    if (options.helpRequested && positionals.empty()) { return Status::Success(); }

    auto status = SetCommand(positionals, options);
    if (!status.Ok()) { return status; }

    if (options.command == CommandType::BENCH && hasCount) {
        if (options.count == 0) {
            return Status::Error(kExitInvalidArgument, "bench --count must be greater than zero");
        }
        options.ioCount = options.count;
        options.count = 0;
    }

    const bool hasRangePart =
        !options.keyPrefix.empty() || options.keyStartSet || options.keyEndSet;
    if (hasRangePart && (options.keyPrefix.empty() || !options.keyStartSet || !options.keyEndSet)) {
        return Status::Error(kExitInvalidArgument,
                             "--prefix, --key-start, and --key-end must be specified together");
    }
    if (options.keyStartSet && options.keyEndSet && options.keyStart > options.keyEnd) {
        return Status::Error(kExitInvalidArgument,
                             "--key-start must be less than or equal to "
                             "--key-end");
    }

    const int keySelectorCount = (hasKey ? 1 : 0) + (hasKeys ? 1 : 0) + (hasKeysFile ? 1 : 0) +
                                 (hasCount ? 1 : 0) + (hasRangePart ? 1 : 0);
    if (keySelectorCount > 1) {
        return Status::Error(kExitInvalidArgument,
                             "--key, --keys, --keys-file, --count, and prefix range are mutually "
                             "exclusive");
    }

    if (options.helpRequested) { return Status::Success(); }

    return Status::Success();
}

}  // namespace kv::bench
