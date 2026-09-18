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
#include <fmt/ranges.h>
#include <unordered_map>
#include "trans_case.h"
#include "trans_runtime.h"

struct TransArgs {
    std::vector<std::string> hosts;
    std::vector<std::string> devices;
    std::vector<std::string> methods;
    TransType type{TransType::ANY};
    std::size_t size{32 * 1024};
    std::size_t number{1024};
    std::size_t nDevice{8};
    std::size_t nIteration{1024};

    TransArgs() = default;
    static void Help(const char* proc)
    {
        fmt::println("Usage: {} [options]", proc ? proc : "trans");
        fmt::println("Options:");
        fmt::println("  -H <host>      Host buffer (can be specified multiple times)");
        fmt::println("  -D <device>    Device buffer (can be specified multiple times)");
        fmt::println("  -M <method>    Transfer method (can be specified multiple times)");
        fmt::println("  -t <type>      Transfer type: H2D or D2H (runs all if not specified)");
        fmt::println("  -s <size>      Transfer size in bytes (default: 32768)");
        fmt::println("  -n <number>    Number of items (default: 1024)");
        fmt::println("  -d <nDevice>   Number of devices (default: 8)");
        fmt::println("  -i <nIter>     Number of iterations (default: 1024)");
        fmt::println("  -h             Show this help message");
    }
    void Show() const
    {
        fmt::println("[[ INPUTS ]]");
        fmt::println("  TransArgs::hosts = [{}]", fmt::join(hosts, ", "));
        fmt::println("  TransArgs::devices = [{}]", fmt::join(devices, ", "));
        fmt::println("  TransArgs::methods = [{}]", fmt::join(methods, ", "));
        fmt::println("  TransArgs::type = {}",
                     type == TransType::ANY ? "ANY" : (type == TransType::H2D ? "H2D" : "D2H"));
        fmt::println("  TransArgs::size = {}", size);
        fmt::println("  TransArgs::number = {}", number);
        fmt::println("  TransArgs::nDevice = {}", nDevice);
        fmt::println("  TransArgs::nIteration = {}", nIteration);
    }
    int32_t Parse(int argc, char const* argv[])
    {
        std::unordered_map<std::string, std::vector<std::string>*> stringLists = {
            {"-H", &hosts  },
            {"-D", &devices},
            {"-M", &methods}
        };
        std::unordered_map<std::string, size_t*> sizeVars = {
            {"-s", &size      },
            {"-n", &number    },
            {"-d", &nDevice   },
            {"-i", &nIteration}
        };
        auto nextArg = [&](int& i, const char* opt) -> const char* {
            if (i + 1 >= argc) {
                fmt::println("Error: missing value for {}", opt);
                return nullptr;
            }
            return argv[++i];
        };
        auto parseSize = [&](const char* s, size_t& out, const char* opt) -> bool {
            try {
                out = std::stoull(s);
                return true;
            } catch (const std::exception& e) {
                fmt::println("Error: invalid number for {}: {}", opt, e.what());
                return false;
            }
        };
        for (int i = 1; i < argc; ++i) {
            std::string opt = argv[i];
            if (opt == "-h") {
                Help(argv[0]);
                return 1;
            }
            if (stringLists.count(opt)) {
                auto val = nextArg(i, opt.c_str());
                if (!val) return -1;
                stringLists[opt]->push_back(val);
                continue;
            }
            if (sizeVars.count(opt)) {
                auto val = nextArg(i, opt.c_str());
                if (!val) return -1;
                if (!parseSize(val, *sizeVars[opt], opt.c_str())) return -1;
                continue;
            }
            if (opt == "-t") {
                auto val = nextArg(i, "-t");
                if (!val) return -1;
                if (std::string(val) == "H2D") {
                    type = TransType::H2D;
                } else if (std::string(val) == "D2H") {
                    type = TransType::D2H;
                } else {
                    fmt::println("Error: unknown type '{}'. Supported: H2D, D2H", val);
                    return -1;
                }
                continue;
            }
            fmt::println("Error: unknown option '{}'", opt);
            return -1;
        }
        return 0;
    }
};

int main(int argc, char const* argv[])
{
    TransArgs args;
    auto ret = args.Parse(argc, argv);
    if (ret != 0) { return ret; }
    args.Show();
    auto& factory = TransCaseFactory::Instance();
    auto cases = factory.Filter(args.type, args.hosts, args.devices, args.methods);
    if (cases.empty()) {
        factory.ShowAllCases();
        return -1;
    }
    TransRuntime runtime;
    TransCtx ctx{args.size, args.number, args.nIteration, args.nDevice};
    for (const auto& c : cases) { c->Run(ctx); }
    return 0;
}
