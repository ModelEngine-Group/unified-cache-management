/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
// 说明：本文件不覆盖 parser_common.h 中的 NormalizeAttrValue、GetUint64Attr、GetBoolAttr。
// 这三个接口目前只有声明、没有定义（同名实现位于 router/config.cc 的匿名命名空间且返回类型不同），
// 直接调用会链接失败，需先在源码侧删除声明或下沉实现后再补充测试。
#include "parser_common.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kv::test {
namespace {

using Attrs = std::unordered_map<std::string, std::string>;

TEST(ParserCommonTest, TrimConfigValueRemovesSurroundingWhitespace)
{
    EXPECT_EQ(TrimConfigValue("  value  "), "value");
    EXPECT_EQ(TrimConfigValue("\t\r\nvalue\r\n\t"), "value");
    EXPECT_EQ(TrimConfigValue("a b"), "a b");
    EXPECT_EQ(TrimConfigValue("  a b  "), "a b");
    EXPECT_EQ(TrimConfigValue(""), "");
    EXPECT_EQ(TrimConfigValue(" \t\r\n "), "");
}

TEST(ParserCommonTest, SplitConfigValueTrimsAndDropsEmptySegments)
{
    EXPECT_EQ(SplitConfigValue("a,b,c", ','), (std::vector<std::string>{"a", "b", "c"}));
    EXPECT_EQ(SplitConfigValue(" a , b ,, c ", ','), (std::vector<std::string>{"a", "b", "c"}));
    EXPECT_EQ(SplitConfigValue(",a,", ','), (std::vector<std::string>{"a"}));
    EXPECT_EQ(SplitConfigValue("", ','), std::vector<std::string>{});
    EXPECT_EQ(SplitConfigValue("a:b::c", ':'), (std::vector<std::string>{"a", "b", "c"}));
}

TEST(ParserCommonTest, ParseConfigUint64SupportsRadixPrefixes)
{
    EXPECT_EQ(ParseConfigUint64("42"), std::uint64_t{42});
    EXPECT_EQ(ParseConfigUint64("0x2A"), std::uint64_t{42});
    EXPECT_EQ(ParseConfigUint64("052"), std::uint64_t{42});
    EXPECT_EQ(ParseConfigUint64(" 42"), std::uint64_t{42});
}

// 不带 fallback 的重载直接暴露 std::stoull 的异常语义。
TEST(ParserCommonTest, ParseConfigUint64WithoutFallbackPropagatesParseErrors)
{
    EXPECT_THROW(ParseConfigUint64("abc"), std::invalid_argument);
    EXPECT_THROW(ParseConfigUint64(""), std::invalid_argument);
    EXPECT_THROW(ParseConfigUint64("18446744073709551616"), std::out_of_range);
}

TEST(ParserCommonTest, ParseConfigUint64WithFallbackRejectsUnconsumedInput)
{
    EXPECT_EQ(ParseConfigUint64("42", 7), std::uint64_t{42});
    EXPECT_EQ(ParseConfigUint64("42abc", 7), std::uint64_t{7});
    EXPECT_EQ(ParseConfigUint64("abc", 7), std::uint64_t{7});
    EXPECT_EQ(ParseConfigUint64("", 7), std::uint64_t{7});
    EXPECT_EQ(ParseConfigUint64("0x", 7), std::uint64_t{7});
    EXPECT_EQ(ParseConfigUint64("18446744073709551616", 7), std::uint64_t{7});
    // 尾随空白不会被消费，按解析失败处理（与前置空白的行为不同）。
    EXPECT_EQ(ParseConfigUint64("42 ", 7), std::uint64_t{7});
}

TEST(ParserCommonTest, ParseConfigUint64WithFallbackStopsAtEmbeddedNul)
{
    // 实现使用 c_str()[consumed] 判断，内嵌 NUL 之后的内容被忽略。
    const std::string value("42\0abc", 6);
    EXPECT_EQ(ParseConfigUint64(value, 7), std::uint64_t{42});
}

TEST(ParserCommonTest, ParseConfigUint32FallsBackWhenOutOfRange)
{
    EXPECT_EQ(ParseConfigUint32("42", 7), std::uint32_t{42});
    EXPECT_EQ(ParseConfigUint32("4294967295", 7), std::uint32_t{4294967295U});
    EXPECT_EQ(ParseConfigUint32("4294967296", 7), std::uint32_t{7});
    EXPECT_EQ(ParseConfigUint32("abc", 7), std::uint32_t{7});
}

TEST(ParserCommonTest, ParseConfigUint16FallsBackWhenOutOfRange)
{
    EXPECT_EQ(ParseConfigUint16("42", 7), std::uint16_t{42});
    EXPECT_EQ(ParseConfigUint16("65535", 7), std::uint16_t{65535U});
    EXPECT_EQ(ParseConfigUint16("65536", 7), std::uint16_t{7});
    EXPECT_EQ(ParseConfigUint16("abc", 7), std::uint16_t{7});
}

TEST(ParserCommonTest, ParseConfigProtocolIsCaseInsensitive)
{
    EXPECT_EQ(ParseConfigProtocol("ub"), Protocol::UB);
    EXPECT_EQ(ParseConfigProtocol("UBOE"), Protocol::UB);
    EXPECT_EQ(ParseConfigProtocol("ubg"), Protocol::UB);
    EXPECT_EQ(ParseConfigProtocol("RoCe"), Protocol::ROCE);
    EXPECT_EQ(ParseConfigProtocol("tcp"), Protocol::TCP);
    // 未知取值默认走 TCP。
    EXPECT_EQ(ParseConfigProtocol("unknown"), Protocol::TCP);
    EXPECT_EQ(ParseConfigProtocol(""), Protocol::TCP);
}

TEST(ParserCommonTest, ParseConfigTransProviderTypeIsCaseInsensitive)
{
    EXPECT_EQ(ParseConfigTransProviderType("fake"), TransProviderType::FAKE);
    EXPECT_EQ(ParseConfigTransProviderType("AIV"), TransProviderType::AIV);
    EXPECT_EQ(ParseConfigTransProviderType("aicpu"), TransProviderType::AICPU);
    EXPECT_EQ(ParseConfigTransProviderType("unknown"), TransProviderType::UNSUPPORTED);
    EXPECT_EQ(ParseConfigTransProviderType(""), TransProviderType::UNSUPPORTED);
}

TEST(ParserCommonTest, ToLowerConvertsAscii)
{
    EXPECT_EQ(ToLower("AbC"), "abc");
    EXPECT_EQ(ToLower(""), "");
    EXPECT_EQ(ToLower("a1-B"), "a1-b");
}

TEST(ParserCommonTest, GetAttrReportsPresenceIncludingEmptyValue)
{
    const Attrs attrs{
        {"key",   "value"},
        {"empty", ""     }
    };
    std::string value;

    ASSERT_TRUE(GetAttr(attrs, "key", value));
    EXPECT_EQ(value, "value");
    ASSERT_TRUE(GetAttr(attrs, "empty", value));
    EXPECT_EQ(value, "");
    EXPECT_FALSE(GetAttr(attrs, "missing", value));
}

TEST(ParserCommonTest, GetAttrWithNamesReturnsFirstPresentKey)
{
    const Attrs attrs{
        {"first", "1"},
        {"empty", "" },
        {"last",  "3"}
    };

    EXPECT_EQ(GetAttr(attrs, {"first", "last"}), "1");
    // 第一个命中的 key 即使值为空也会立即返回。
    EXPECT_EQ(GetAttr(attrs, {"empty", "last"}), "");
    EXPECT_EQ(GetAttr(attrs, {"missing", "last"}), "3");
    EXPECT_EQ(GetAttr(attrs, {"missing", "absent"}), "");
}

TEST(ParserCommonTest, GetConfigAndEndpointAttrUseTheirAttrMaps)
{
    TransportConfig config;
    config.attrs["config_key"] = "config_value";

    NodeEndpoint endpoint;
    endpoint.attrs["endpoint_key"] = "endpoint_value";

    EXPECT_EQ(GetConfigAttr(config, {"missing", "config_key"}), "config_value");
    EXPECT_EQ(GetEndpointAttr(&endpoint, {"endpoint_key"}), "endpoint_value");
    EXPECT_EQ(GetEndpointAttr(nullptr, {"endpoint_key"}), "");
}

TEST(ParserCommonTest, ApplyTransportBufferConfigFieldHandlesAliases)
{
    TransportConfig config;

    EXPECT_TRUE(ApplyTransportBufferConfigField(config, "sendBufferSlotSize", "1024"));
    EXPECT_EQ(config.sendBufferSlotSize, std::size_t{1024});
    EXPECT_TRUE(ApplyTransportBufferConfigField(config, "send_buffer_slot_num", "8"));
    EXPECT_EQ(config.sendBufferSlotNum, std::size_t{8});
    EXPECT_TRUE(ApplyTransportBufferConfigField(config, "ioBuffer.sendBufferSlotSize", "2048"));
    EXPECT_EQ(config.sendBufferSlotSize, std::size_t{2048});
    EXPECT_TRUE(ApplyTransportBufferConfigField(config, "io_buffer.send_buffer_slot_num", "9"));
    EXPECT_EQ(config.sendBufferSlotNum, std::size_t{9});
    EXPECT_TRUE(ApplyTransportBufferConfigField(config, "flagBufferSlotSize", "71"));
    EXPECT_EQ(config.flagBufferSlotSize, std::size_t{71});
    EXPECT_TRUE(ApplyTransportBufferConfigField(config, "flag_buffer_slot_num", "16"));
    EXPECT_EQ(config.flagBufferSlotNum, std::size_t{16});
    EXPECT_TRUE(ApplyTransportBufferConfigField(config, "io_buffer.flag_buffer_slot_size", "72"));
    EXPECT_EQ(config.flagBufferSlotSize, std::size_t{72});

    EXPECT_FALSE(ApplyTransportBufferConfigField(config, "unknown", "1"));
}

TEST(ParserCommonTest, ApplyTransportIoNumConfigFieldHandlesAliases)
{
    TransportConfig config;

    EXPECT_TRUE(ApplyTransportIoNumConfigField(config, "batchLoadIoNum", "11"));
    EXPECT_EQ(config.batchLoadIoNum, std::size_t{11});
    EXPECT_TRUE(ApplyTransportIoNumConfigField(config, "batch_load_io_num", "12"));
    EXPECT_EQ(config.batchLoadIoNum, std::size_t{12});
    EXPECT_TRUE(ApplyTransportIoNumConfigField(config, "batchStoreIoNum", "13"));
    EXPECT_EQ(config.batchStoreIoNum, std::size_t{13});
    EXPECT_TRUE(ApplyTransportIoNumConfigField(config, "batch_store_io_num", "14"));
    EXPECT_EQ(config.batchStoreIoNum, std::size_t{14});
    EXPECT_TRUE(ApplyTransportIoNumConfigField(config, "deleteIoNum", "15"));
    EXPECT_EQ(config.deleteIoNum, std::size_t{15});
    EXPECT_TRUE(ApplyTransportIoNumConfigField(config, "query_io_num", "16"));
    EXPECT_EQ(config.queryIoNum, std::size_t{16});

    EXPECT_FALSE(ApplyTransportIoNumConfigField(config, "batch_load_io", "1"));
}

TEST(ParserCommonTest, ApplyTransportCompletionConfigFieldHandlesAliases)
{
    TransportConfig config;

    EXPECT_TRUE(ApplyTransportCompletionConfigField(config, "completionPollSpinLimit", "5"));
    EXPECT_EQ(config.completionPollSpinLimit, std::size_t{5});
    EXPECT_TRUE(ApplyTransportCompletionConfigField(config, "completion_poll_spin_limit", "6"));
    EXPECT_EQ(config.completionPollSpinLimit, std::size_t{6});

    EXPECT_FALSE(ApplyTransportCompletionConfigField(config, "completion_poll_limit", "7"));
}

TEST(ParserCommonTest, ApplyTransportProviderConfigFieldHandlesAliases)
{
    for (const char* key :
         {"providerBackend", "provider_backend", "transProviderBackend", "trans_provider_backend",
          "providerType", "provider_type", "transProviderType", "trans_provider_type"}) {
        TransportConfig config;
        EXPECT_TRUE(ApplyTransportProviderConfigField(config, key, "fake")) << key;
        EXPECT_EQ(config.providerType, TransProviderType::FAKE) << key;
    }

    TransportConfig config;
    EXPECT_TRUE(ApplyTransportProviderConfigField(config, "provider_type", "AIV"));
    EXPECT_EQ(config.providerType, TransProviderType::AIV);
    EXPECT_TRUE(ApplyTransportProviderConfigField(config, "provider_type", "unknown"));
    EXPECT_EQ(config.providerType, TransProviderType::UNSUPPORTED);
    EXPECT_FALSE(ApplyTransportProviderConfigField(config, "provider", "FAKE"));
}

TEST(ParserCommonTest, ApplyTransportDeviceConfigFieldHandlesAliases)
{
    TransportConfig config;

    EXPECT_TRUE(ApplyTransportDeviceConfigField(config, "deviceId", "3"));
    EXPECT_EQ(config.deviceId, std::int32_t{3});
    EXPECT_TRUE(ApplyTransportDeviceConfigField(config, "device_id", "4"));
    EXPECT_EQ(config.deviceId, std::int32_t{4});

    EXPECT_FALSE(ApplyTransportDeviceConfigField(config, "device", "5"));
}

TEST(ParserCommonTest, ParseTransportEndpointSupportsShorthandForm)
{
    const auto endpoint = ParseTransportEndpoint("10.0.0.1:1234:ROCE");

    EXPECT_EQ(endpoint.ip, "10.0.0.1");
    EXPECT_EQ(endpoint.port, std::uint16_t{1234});
    EXPECT_EQ(endpoint.protocol, Protocol::ROCE);
}

TEST(ParserCommonTest, ParseTransportEndpointShorthandDefaultsRemainingFields)
{
    const auto withPort = ParseTransportEndpoint("10.0.0.1:2345");
    EXPECT_EQ(withPort.ip, "10.0.0.1");
    EXPECT_EQ(withPort.port, std::uint16_t{2345});
    // NodeEndpoint 的默认协议是 ROCE。
    EXPECT_EQ(withPort.protocol, Protocol::ROCE);

    const auto hostOnly = ParseTransportEndpoint("10.0.0.1");
    EXPECT_EQ(hostOnly.ip, "10.0.0.1");
    EXPECT_EQ(hostOnly.port, std::uint16_t{0});
}

TEST(ParserCommonTest, ParseTransportEndpointSupportsKeyValueForm)
{
    const auto endpoint = ParseTransportEndpoint(
        "ip=10.0.0.1,port=1234,protocol=UB,hca_name=eth0,hca_port=2,numa_node=1,custom=xyz");

    EXPECT_EQ(endpoint.ip, "10.0.0.1");
    EXPECT_EQ(endpoint.port, std::uint16_t{1234});
    EXPECT_EQ(endpoint.protocol, Protocol::UB);
    EXPECT_EQ(endpoint.hcaName, "eth0");
    EXPECT_EQ(endpoint.hcaPort, std::uint8_t{2});
    EXPECT_EQ(endpoint.numaNode, std::int32_t{1});
    ASSERT_EQ(endpoint.attrs.count("custom"), 1U);
    EXPECT_EQ(endpoint.attrs.at("custom"), "xyz");
}

TEST(ParserCommonTest, ParseTransportEndpointAcceptsLegacyAliases)
{
    const auto endpoint = ParseTransportEndpoint("local.comm_id=1.2.3.4,hcaPort=3,numaNode=2");

    EXPECT_EQ(endpoint.ip, "1.2.3.4");
    EXPECT_EQ(endpoint.hcaPort, std::uint8_t{3});
    EXPECT_EQ(endpoint.numaNode, std::int32_t{2});
}

TEST(ParserCommonTest, ParseTransportEndpointSkipsSegmentsWithoutValue)
{
    const auto endpoint = ParseTransportEndpoint("ip=1.1.1.1,bogus,port=80");

    EXPECT_EQ(endpoint.ip, "1.1.1.1");
    EXPECT_EQ(endpoint.port, std::uint16_t{80});
}

// 当前实现对非法端口没有兜底，异常会直接向上传播。
TEST(ParserCommonTest, ParseTransportEndpointShorthandPropagatesInvalidPort)
{
    EXPECT_THROW(ParseTransportEndpoint("10.0.0.1:notaport"), std::invalid_argument);
}

}  // namespace
}  // namespace kv::test
