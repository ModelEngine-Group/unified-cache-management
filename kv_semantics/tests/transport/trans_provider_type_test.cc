// The kv-test configuration path and the ASU store must agree on the provider
// names: a transport named in a configuration file silently falls back to
// UNSUPPORTED when the generic parser does not know the name.

#include <gtest/gtest.h>
#include <string>
#include "parser_common.h"

namespace kv::test {
namespace {

TEST(ParseConfigTransProviderTypeTest, AcceptsTheCpuProviderName)
{
    EXPECT_EQ(TransProviderType::CPU, ParseConfigTransProviderType("CPU"));
    EXPECT_EQ(TransProviderType::CPU, ParseConfigTransProviderType("cpu"));
}

TEST(ParseConfigTransProviderTypeTest, KeepsTheExistingProviderNames)
{
    EXPECT_EQ(TransProviderType::FAKE, ParseConfigTransProviderType("fake"));
    EXPECT_EQ(TransProviderType::AICPU, ParseConfigTransProviderType("AICPU"));
    EXPECT_EQ(TransProviderType::AIV, ParseConfigTransProviderType("aiv"));
    EXPECT_EQ(TransProviderType::UNSUPPORTED, ParseConfigTransProviderType("unknown"));
}

}  // namespace
}  // namespace kv::test
