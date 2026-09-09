#include "arg_parser.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace kv::bench {
namespace {

Status ParseArguments(std::vector<std::string> arguments, CommandOptions& options)
{
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) { argv.emplace_back(argument.data()); }
    return ArgParser{}.Parse(static_cast<int>(argv.size()), argv.data(), options);
}

TEST(ArgParserTest, BenchCountSetsTotalIoCount)
{
    CommandOptions options;
    auto status = ParseArguments({"kv-test", "bench", "batch-store", "--count", "100"}, options);

    ASSERT_TRUE(status.Ok()) << status.message;
    EXPECT_EQ(options.command, CommandType::BENCH);
    EXPECT_EQ(options.ioCount, 100U);
    EXPECT_EQ(options.count, 0U);
}

TEST(ArgParserTest, NonBenchCountKeepsKeyGenerationCount)
{
    CommandOptions options;
    auto status = ParseArguments({"kv-test", "store", "--count", "100"}, options);

    ASSERT_TRUE(status.Ok()) << status.message;
    EXPECT_EQ(options.command, CommandType::STORE);
    EXPECT_EQ(options.count, 100U);
    EXPECT_EQ(options.ioCount, 0U);
}

TEST(ArgParserTest, BenchCountMustBePositive)
{
    CommandOptions options;
    auto status = ParseArguments({"kv-test", "bench", "store", "--count", "0"}, options);

    EXPECT_FALSE(status.Ok());
    EXPECT_EQ(status.message, "bench --count must be greater than zero");
}

}  // namespace
}  // namespace kv::bench
