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
#include "view/m800_view_codec.h"
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <unordered_map>
#include "view/m800_view_common.h"

namespace kv::test {
namespace {

GlobalView MakeView(KvNsId nsId, std::uint64_t epoch)
{
    GlobalView view;
    view.viewEpoch = epoch;
    view.viewId = nsId;
    return view;
}

NodeEndpoint MakeEndpoint(const std::string& ip = "10.0.0.1")
{
    NodeEndpoint endpoint;
    endpoint.protocol = Protocol::UB;
    endpoint.port = 6000;
    endpoint.ip = ip;
    endpoint.attrs["protocol"] = "uboe";
    return endpoint;
}

std::string ValidText()
{
    return "globalEpoch=123\n"
           "nsCount=1\n"
           "ns=1\n"
           "viewEpoch=123\n"
           "viewId=1\n"
           "asuIds=10\n"
           "asuInfo.10=protocol=uboe,port=6000,local.comm_id=10.0.0.1\n";
}

// 测试场景：编码 localKvNsId 默认为 0 的节点，并分别解码省略该字段和显式设为 0 的文本。
// 预期结果：编码结果与 ValidText 完全一致且不包含 localKvNsId. 字段；两种解码结果的 localKvNsId
// 均为 0，显式零值再次编码也不输出该字段。
TEST(M800ViewCodecTest, DefaultZeroLocalKvNsIdPreservesWireFormat)
{
    EXPECT_STREQ(M800ViewTextContract::kLocalKvNsIdFieldPrefix, "localKvNsId.");
    auto view = MakeView(1, 123);
    view.asuMap.emplace(10, NodeInfo{{MakeEndpoint()}});
    auto universe = CreateViewUniverse(
        {
            {1, std::move(view)}
    },
        123);

    std::string encoded;
    ASSERT_TRUE(M800ViewCodec::Encode(*universe, encoded).ok());
    EXPECT_EQ(encoded, ValidText());
    EXPECT_EQ(encoded.find("localKvNsId."), std::string::npos);

    auto decoded = CreateViewUniverse();
    ASSERT_TRUE(M800ViewCodec::Decode(encoded, *decoded).ok());
    GlobalView actual;
    ASSERT_TRUE(decoded->GetView(1, actual).ok());
    EXPECT_EQ(actual.asuMap.at(10).localKvNsId, 0);

    auto explicitZero = CreateViewUniverse();
    ASSERT_TRUE(M800ViewCodec::Decode(ValidText() + "localKvNsId.10=0\n", *explicitZero).ok());
    ASSERT_TRUE(explicitZero->GetView(1, actual).ok());
    EXPECT_EQ(actual.asuMap.at(10).localKvNsId, 0);
    ASSERT_TRUE(M800ViewCodec::Encode(*explicitZero, encoded).ok());
    EXPECT_EQ(encoded, ValidText());
}

// 测试场景：对 localKvNsId 为 uint32_t 最大值且包含两个端点的节点执行编码、解码和再次编码。
// 预期结果：编码包含首个端点及 localKvNsId.10=4294967295；解码保留最大 ID 和两个 IP
// 的顺序，再编码文本与首次完全一致。
TEST(M800ViewCodecTest, NonZeroLocalKvNsIdRoundTripsWithMultipleEndpoints)
{
    auto view = MakeView(1, 123);
    NodeInfo info{
        {MakeEndpoint("10.0.0.1"), MakeEndpoint("10.0.0.2")}
    };
    info.localKvNsId = std::numeric_limits<KvNsId>::max();
    view.asuMap.emplace(10, std::move(info));
    auto universe = CreateViewUniverse(
        {
            {1, std::move(view)}
    },
        123);

    std::string encoded;
    ASSERT_TRUE(M800ViewCodec::Encode(*universe, encoded).ok());
    EXPECT_NE(encoded.find("local.comm_id=10.0.0.1;protocol=uboe"), std::string::npos);
    EXPECT_NE(encoded.find("localKvNsId.10=4294967295\n"), std::string::npos);

    auto decoded = CreateViewUniverse();
    ASSERT_TRUE(M800ViewCodec::Decode(encoded, *decoded).ok()) << encoded;
    GlobalView actual;
    ASSERT_TRUE(decoded->GetView(1, actual).ok());
    const auto& actualInfo = actual.asuMap.at(10);
    EXPECT_EQ(actualInfo.localKvNsId, std::numeric_limits<KvNsId>::max());
    ASSERT_EQ(actualInfo.endpoints.size(), 2);
    EXPECT_EQ(actualInfo.endpoints[0].ip, "10.0.0.1");
    EXPECT_EQ(actualInfo.endpoints[1].ip, "10.0.0.2");

    std::string reencoded;
    ASSERT_TRUE(M800ViewCodec::Encode(*decoded, reencoded).ok());
    EXPECT_EQ(reencoded, encoded);
}

// 测试场景：按 20、10 的顺序插入两个带非零 localKvNsId 的 ASU 后编码视图。
// 预期结果：ASU 按 10、20 升序输出，每条 localKvNsId 紧跟对应的
// asuInfo，完整编码文本与预期逐字一致。
TEST(M800ViewCodecTest, LocalKvNsIdEncodingIsOrderedAndAdjacentToItsAsuInfo)
{
    auto view = MakeView(1, 123);
    NodeInfo node20{{MakeEndpoint("10.0.0.20")}};
    node20.localKvNsId = 20;
    NodeInfo node10{{MakeEndpoint("10.0.0.10")}};
    node10.localKvNsId = 10;
    view.asuMap.emplace(20, std::move(node20));
    view.asuMap.emplace(10, std::move(node10));
    auto universe = CreateViewUniverse(
        {
            {1, std::move(view)}
    },
        123);

    std::string encoded;
    ASSERT_TRUE(M800ViewCodec::Encode(*universe, encoded).ok());
    const std::string expected =
        "globalEpoch=123\n"
        "nsCount=1\n"
        "ns=1\n"
        "viewEpoch=123\n"
        "viewId=1\n"
        "asuIds=10,20\n"
        "asuInfo.10=protocol=uboe,port=6000,local.comm_id=10.0.0.10\n"
        "localKvNsId.10=10\n"
        "asuInfo.20=protocol=uboe,port=6000,local.comm_id=10.0.0.20\n"
        "localKvNsId.20=20\n";
    EXPECT_EQ(encoded, expected);
}

// 测试场景：向合法文本追加重复、指向未声明 ASU、键名错误、空值、非十进制、负数、带后缀或超范围的
// localKvNsId 字段，并解码到已有视图。 预期结果：每种输入均返回 INTERNAL_ERROR，原 global epoch
// 55、唯一 ASU 7 的 localKvNsId 77 及端点 10.0.0.7 均保持不变。
TEST(M800ViewCodecTest, DecodeRejectsInvalidLocalKvNsIdWithoutChangingOutput)
{
    const std::vector<std::string> invalidSuffixes = {
        "localKvNsId.10=1\nlocalKvNsId.10=2\n",
        "localKvNsId.11=1\n",
        "localKvNsId.10x=1\n",
        "localKvNsId.10=\n",
        "localKvNsId.10=abc\n",
        "localKvNsId.10=+1\n",
        "localKvNsId.10=-1\n",
        "localKvNsId.10=0x1\n",
        "localKvNsId.10=1suffix\n",
        "localKvNsId.10=4294967296\n",
    };
    auto baselineView = MakeView(7, 55);
    NodeInfo baselineInfo{{MakeEndpoint("10.0.0.7")}};
    baselineInfo.localKvNsId = 77;
    baselineView.asuMap.emplace(7, std::move(baselineInfo));
    auto baseline = CreateViewUniverse(
        {
            {7, std::move(baselineView)}
    },
        55);

    for (const auto& suffix : invalidSuffixes) {
        const auto input = ValidText() + suffix;
        const auto status = M800ViewCodec::Decode(input, *baseline);
        EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR) << input;
        EXPECT_EQ(baseline->GetGlobalEpoch(), 55) << input;
        GlobalView actual;
        ASSERT_TRUE(baseline->GetView(7, actual).ok()) << input;
        ASSERT_EQ(actual.asuMap.size(), 1) << input;
        EXPECT_EQ(actual.asuMap.at(7).localKvNsId, 77) << input;
        ASSERT_EQ(actual.asuMap.at(7).endpoints.size(), 1) << input;
        EXPECT_EQ(actual.asuMap.at(7).endpoints.front().ip, "10.0.0.7") << input;
    }
}

// 测试场景：将 localKvNsId.10 放在对应 asuInfo.10 之前，或用其他字段将二者隔开后解码。
// 预期结果：两种非相邻布局均返回 INTERNAL_ERROR，输出视图宇宙保持为空且 global epoch 为 0。
TEST(M800ViewCodecTest, DecodeRejectsNonAdjacentLocalKvNsId)
{
    auto beforeAsuInfo = ValidText();
    const auto pos = beforeAsuInfo.find("asuInfo.10=");
    ASSERT_NE(pos, std::string::npos);
    beforeAsuInfo.insert(pos, "localKvNsId.10=1\n");
    const auto separated = ValidText() + "futureField=value\nlocalKvNsId.10=1\n";
    auto baseline = CreateViewUniverse();

    for (const auto& input : {beforeAsuInfo, separated}) {
        EXPECT_EQ(M800ViewCodec::Decode(input, *baseline).code, StatusCode::INTERNAL_ERROR)
            << input;
        EXPECT_TRUE(baseline->Empty());
        EXPECT_EQ(baseline->GetGlobalEpoch(), 0);
    }
}

// 测试场景：将合法视图文本中的端点端口改为 0 后解码。
// 预期结果：返回 INTERNAL_ERROR，输出视图宇宙保持为空。
TEST(M800ViewCodecTest, DecodeRejectsZeroEndpointPortWithoutChangingOutput)
{
    auto input = ValidText();
    const auto pos = input.find("port=6000");
    ASSERT_NE(pos, std::string::npos);
    input.replace(pos, 9, "port=0");
    auto universe = CreateViewUniverse();
    EXPECT_EQ(M800ViewCodec::Decode(input, *universe).code, StatusCode::INTERNAL_ERROR);
    EXPECT_TRUE(universe->Empty());
}

// 测试场景：编码包含端口为 0 的端点的视图，输出字符串预置为 unchanged。
// 预期结果：返回 INTERNAL_ERROR，输出字符串仍为 unchanged，不写入部分编码内容。
TEST(M800ViewCodecTest, EncodeRejectsZeroEndpointPortWithoutChangingOutput)
{
    auto view = MakeView(1, 123);
    auto endpoint = MakeEndpoint();
    endpoint.port = 0;
    view.asuMap.emplace(10, NodeInfo{{endpoint}});
    auto universe = CreateViewUniverse(
        {
            {1, std::move(view)}
    },
        123);
    std::string output = "unchanged";
    EXPECT_EQ(M800ViewCodec::Encode(*universe, output).code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(output, "unchanged");
}

// 测试场景：以无序方式存放两个命名空间，并编码含 IPv4、IPv6、NUMA/HCA
// 信息和乱序属性的端点后再解码、重编码。 预期结果：命名空间按 ID 升序、asuIds
// 和属性按排序结果输出；解码保留 global epoch 123、两个命名空间及 IPv6
// 端点各字段，重编码与首次相同。
TEST(M800ViewCodecTest, EncodeIsDeterministicAndRoundTripsEndpoints)
{
    std::unordered_map<KvNsId, GlobalView> views;
    auto first = MakeView(2, 123);
    first.asuMap.emplace(20, NodeInfo{{MakeEndpoint("10.0.0.20")}});
    auto second = MakeView(1, 123);
    NodeEndpoint endpoint = MakeEndpoint("2001:db8::1");
    endpoint.numaNode = 2;
    endpoint.hcaName = "mlx5_0";
    endpoint.hcaPort = 2;
    endpoint.attrs["z_key"] = "last";
    endpoint.attrs["a_key"] = "first";
    second.asuMap.emplace(20, NodeInfo{{MakeEndpoint("10.0.0.2")}});
    second.asuMap.emplace(10, NodeInfo{{endpoint}});
    views.emplace(2, std::move(first));
    views.emplace(1, std::move(second));
    auto universe = CreateViewUniverse(std::move(views), 123);

    std::string encoded;
    ASSERT_TRUE(M800ViewCodec::Encode(*universe, encoded).ok());
    EXPECT_LT(encoded.find("ns=1\n"), encoded.find("ns=2\n"));
    EXPECT_LT(encoded.find("asuIds=10,20\n"), encoded.find("ns=2\n"));
    EXPECT_NE(encoded.find("a_key=first,z_key=last"), std::string::npos);

    auto decoded = CreateViewUniverse();
    ASSERT_TRUE(M800ViewCodec::Decode(encoded, *decoded).ok());
    EXPECT_EQ(decoded->GetGlobalEpoch(), 123);
    EXPECT_EQ(decoded->GetAllNsIds().size(), 2);
    GlobalView view;
    ASSERT_TRUE(decoded->GetView(1, view).ok());
    ASSERT_EQ(view.asuMap.at(10).endpoints.size(), 1);
    const auto& actual = view.asuMap.at(10).endpoints.front();
    EXPECT_EQ(actual.protocol, Protocol::UB);
    EXPECT_EQ(actual.port, 6000);
    EXPECT_EQ(actual.ip, "2001:db8::1");
    EXPECT_EQ(actual.numaNode, 2);
    EXPECT_EQ(actual.hcaName, "mlx5_0");
    EXPECT_EQ(actual.hcaPort, 2);
    EXPECT_EQ(actual.attrs.at("a_key"), "first");

    std::string reencoded;
    ASSERT_TRUE(M800ViewCodec::Encode(*decoded, reencoded).ok());
    EXPECT_EQ(reencoded, encoded);
}

// 测试场景：分别将合法文本转换为 CRLF，以及删除最后一个 LF，并在已有 epoch 55、namespace 7
// 的输出对象上解码。 预期结果：两种输入均返回 INTERNAL_ERROR，原 epoch 和 namespace 完整保持不变。
TEST(M800ViewCodecTest, DecodeRejectsCrlfAndMissingFinalNewlineWithoutChangingOutput)
{
    std::string crlf;
    for (const char ch : ValidText()) {
        if (ch == '\n') {
            crlf += "\r\n";
        } else {
            crlf.push_back(ch);
        }
    }
    auto missingFinalLf = ValidText();
    missingFinalLf.pop_back();

    for (const auto& input : {crlf, missingFinalLf}) {
        auto baseline = CreateViewUniverse(
            {
                {7, MakeView(7, 55)}
        },
            55);
        EXPECT_EQ(M800ViewCodec::Decode(input, *baseline).code, StatusCode::INTERNAL_ERROR);
        EXPECT_EQ(baseline->GetGlobalEpoch(), 55);
        ASSERT_EQ(baseline->GetAllNsIds().size(), 1);
        EXPECT_EQ(baseline->GetAllNsIds().front(), 7);
    }
}

// 测试场景：在已有 epoch 55、namespace 7 的 Universe 上解码 `globalEpoch=1, nsCount=0` 文本。
// 预期结果：Codec 按纯解析和 ReplaceAll 契约成功清空视图并把 epoch 设为 1；Slave
// 发布前另行拒绝空快照。
TEST(M800ViewCodecTest, DecodeZeroNamespaceSnapshotUsesReplaceAllSemantics)
{
    auto universe = CreateViewUniverse(
        {
            {7, MakeView(7, 55)}
    },
        55);

    ASSERT_TRUE(M800ViewCodec::Decode("globalEpoch=1\nnsCount=0\n", *universe).ok());

    EXPECT_TRUE(universe->Empty());
    EXPECT_EQ(universe->GetGlobalEpoch(), 1);
}

// 测试场景：解码空文本、非法 epoch、字段或 ID
// 不一致、非法端口或协议、未知外层字段及异常控制字符等多组畸形输入，输出对象预置 epoch 55
// 和命名空间 7。 预期结果：所有输入均返回 INTERNAL_ERROR，且原 global epoch 55 和唯一命名空间 7
// 不被修改。
TEST(M800ViewCodecTest, DecodeRejectsMalformedInputWithoutChangingOutput)
{
    std::vector<std::string> invalidInputs;
    invalidInputs.emplace_back("");
    invalidInputs.emplace_back("globalEpoch=1\nnsCount=0\nns=1\n");
    invalidInputs.emplace_back("globalEpoch=-1\nnsCount=0\n");
    invalidInputs.emplace_back("globalEpoch=+1\nnsCount=0\n");
    invalidInputs.emplace_back("globalEpoch=0x10\nnsCount=0\n");
    invalidInputs.emplace_back("globalEpoch=12x\nnsCount=0\n");
    invalidInputs.emplace_back("globalEpoch=18446744073709551616\nnsCount=0\n");

    const std::vector<std::pair<std::string, std::string>> replacements = {
        {"nsCount=1",              "nsCount=2"                     },
        {"viewEpoch=123",          "viewEpoch=122"                 },
        {"viewId=1",               "viewId=2"                      },
        {"viewId=1",               "viewId=1\nviewId=1"            },
        {"asuIds=10",              "asuIds=10,11"                  },
        {"asuInfo.10=",            "asuInfo.11="                   },
        {"port=6000",              "port=65536"                    },
        {"port=6000",              "port=6000,numa_node=-1"        },
        {"protocol=uboe",          "protocol=unknown"              },
        {"local.comm_id=10.0.0.1", "local.comm_id="                },
        {"asuInfo.10",             "asuInfo.10x"                   },
        {"port=6000",              "port=60=00"                    },
        {"asuIds=10",              "this-is-not-a-field\nasuIds=10"},
        {"asuIds=10",              "=value\nasuIds=10"             },
        {"asuIds=10",              "futureField=a=b\nasuIds=10"    },
        {"asuIds=10",              "futureField=a,b\nasuIds=10"    },
        {"ns=1\n",                 "ns=1\nns=1\n"                  },
    };
    for (const auto& [from, to] : replacements) {
        auto input = ValidText();
        const auto pos = input.find(from);
        ASSERT_NE(pos, std::string::npos) << from;
        input.replace(pos, from.size(), to);
        invalidInputs.emplace_back(std::move(input));
    }
    auto embeddedCr = ValidText();
    embeddedCr.insert(embeddedCr.find("asuIds"), "\r");
    invalidInputs.emplace_back(embeddedCr);
    auto embeddedNul = ValidText();
    embeddedNul.push_back('\0');
    invalidInputs.emplace_back(embeddedNul);
    auto isolatedCr = ValidText();
    isolatedCr.pop_back();
    isolatedCr.push_back('\r');
    invalidInputs.emplace_back(isolatedCr);

    auto baseline = CreateViewUniverse(
        {
            {7, MakeView(7, 55)}
    },
        55);
    for (const auto& input : invalidInputs) {
        auto status = M800ViewCodec::Decode(input, *baseline);
        EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR) << input;
        EXPECT_EQ(baseline->GetGlobalEpoch(), 55) << input;
        ASSERT_EQ(baseline->GetAllNsIds().size(), 1) << input;
        EXPECT_EQ(baseline->GetAllNsIds().front(), 7) << input;
    }
}

// 测试场景：分别在命名空间块内的 asuIds 前和命名空间块外的 ns=1 前插入 futureField 未知字段后解码。
// 预期结果：块内和块外未知字段均返回 INTERNAL_ERROR，已有 epoch 55 和 namespace 7 均保持不变。
TEST(M800ViewCodecTest, UnknownFieldsAreRejectedInsideAndOutsideNamespace)
{
    auto insideBlock = ValidText();
    insideBlock.insert(insideBlock.find("asuIds="), "futureField=anything\n");
    auto outsideBlock = ValidText();
    outsideBlock.insert(outsideBlock.find("ns=1\n"), "futureField=anything\n");

    for (const auto& input : {insideBlock, outsideBlock}) {
        auto unchanged = CreateViewUniverse(
            {
                {7, MakeView(7, 55)}
        },
            55);
        EXPECT_EQ(M800ViewCodec::Decode(input, *unchanged).code, StatusCode::INTERNAL_ERROR);
        EXPECT_EQ(unchanged->GetGlobalEpoch(), 55);
        ASSERT_EQ(unchanged->GetAllNsIds().size(), 1);
        EXPECT_EQ(unchanged->GetAllNsIds().front(), 7);
    }
}

// 测试场景：分别编码 viewEpoch 与全局 epoch
// 不一致的视图，以及端点属性值含逗号的视图，输出字符串预置为 unchanged。 预期结果：两次编码均返回
// INTERNAL_ERROR，且输出字符串始终保持 unchanged。
TEST(M800ViewCodecTest, EncodeRejectsInconsistentViewAndInvalidScalar)
{
    auto invalid = MakeView(1, 10);
    invalid.viewEpoch = 9;
    auto universe = CreateViewUniverse(
        {
            {1, invalid}
    },
        10);
    std::string output = "unchanged";
    EXPECT_EQ(M800ViewCodec::Encode(*universe, output).code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(output, "unchanged");

    auto valid = MakeView(1, 10);
    auto endpoint = MakeEndpoint();
    endpoint.attrs["bad"] = "a,b";
    valid.asuMap.emplace(1, NodeInfo{{endpoint}});
    universe = CreateViewUniverse(
        {
            {1, valid}
    },
        10);
    EXPECT_EQ(M800ViewCodec::Encode(*universe, output).code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(output, "unchanged");
}

// 测试场景：编码 65 个命名空间、每个 ASU 含 5 个端点的视图，再解码编码结果。
// 预期结果：nsCount 为 64，仅保留前 64 个命名空间且不输出 ns=65；解码后命名空间 1 的 ASU 仅保留 4
// 个端点。
TEST(M800ViewCodecTest, NamespaceAndEndpointLimitsAreTruncatedDeterministically)
{
    std::unordered_map<KvNsId, GlobalView> views;
    for (KvNsId nsId = 1; nsId <= 65; ++nsId) {
        auto view = MakeView(nsId, 99);
        NodeInfo info;
        for (int index = 0; index < 5; ++index) {
            info.endpoints.emplace_back(MakeEndpoint("10.0.0." + std::to_string(index + 1)));
        }
        view.asuMap.emplace(1, std::move(info));
        views.emplace(nsId, std::move(view));
    }
    auto universe = CreateViewUniverse(std::move(views), 99);
    std::string encoded;
    ASSERT_TRUE(M800ViewCodec::Encode(*universe, encoded).ok());
    EXPECT_NE(encoded.find("nsCount=64\n"), std::string::npos);
    EXPECT_EQ(encoded.find("ns=65\n"), std::string::npos);

    auto decoded = CreateViewUniverse();
    ASSERT_TRUE(M800ViewCodec::Decode(encoded, *decoded).ok());
    EXPECT_EQ(decoded->GetAllNsIds().size(), 64);
    GlobalView view;
    ASSERT_TRUE(decoded->GetView(1, view).ok());
    EXPECT_EQ(view.asuMap.at(1).endpoints.size(), 4);
}

// 测试场景：直接解码分别包含 65 个命名空间、单 ASU 5 个端点、4097 个 ASU 的三种超限文本。
// 预期结果：依次仅保留 64 个命名空间、前 4 个端点及前 4096 个 ASU；第 64 个命名空间、端点 10.0.0.4
// 和 ASU 4096 的 localKvNsId 均正确保留，超限项被丢弃。
TEST(M800ViewCodecTest, DecodeDirectlyTruncatesOversizedWireInput)
{
    const std::string endpoint = "protocol=uboe,port=6000,local.comm_id=10.0.0.1";
    std::string text = "globalEpoch=99\nnsCount=65\n";
    for (KvNsId nsId = 1; nsId <= 65; ++nsId) {
        text += "ns=" + std::to_string(nsId) + "\nviewEpoch=99\nviewId=" + std::to_string(nsId) +
                "\nasuIds=1\nasuInfo.1=" + endpoint + "\n";
        if (nsId >= 64) { text += "localKvNsId.1=" + std::to_string(nsId) + "\n"; }
    }
    auto decoded = CreateViewUniverse();
    ASSERT_TRUE(M800ViewCodec::Decode(text, *decoded).ok());
    EXPECT_EQ(decoded->GetAllNsIds().size(), 64);
    GlobalView view;
    EXPECT_TRUE(decoded->GetView(1, view).ok());
    EXPECT_EQ(decoded->GetView(65, view).code, StatusCode::NOT_FOUND);
    ASSERT_TRUE(decoded->GetView(64, view).ok());
    EXPECT_EQ(view.asuMap.at(1).localKvNsId, 64);

    text = "globalEpoch=99\nnsCount=1\nns=1\nviewEpoch=99\nviewId=1\nasuIds=1\nasuInfo.1=";
    for (int index = 0; index < 5; ++index) {
        if (index != 0) { text.push_back(';'); }
        text += "protocol=uboe,port=6000,local.comm_id=10.0.0." + std::to_string(index + 1);
    }
    text.push_back('\n');
    text += "localKvNsId.1=9\n";
    decoded = CreateViewUniverse();
    ASSERT_TRUE(M800ViewCodec::Decode(text, *decoded).ok());
    ASSERT_TRUE(decoded->GetView(1, view).ok());
    EXPECT_EQ(view.asuMap.at(1).localKvNsId, 9);
    ASSERT_EQ(view.asuMap.at(1).endpoints.size(), 4);
    EXPECT_EQ(view.asuMap.at(1).endpoints.back().ip, "10.0.0.4");

    text = "globalEpoch=99\nnsCount=1\nns=1\nviewEpoch=99\nviewId=1\nasuIds=";
    for (NodeId nodeId = 1; nodeId <= 4097; ++nodeId) {
        if (nodeId != 1) { text.push_back(','); }
        text += std::to_string(nodeId);
    }
    text.push_back('\n');
    for (NodeId nodeId = 1; nodeId <= 4097; ++nodeId) {
        text += "asuInfo." + std::to_string(nodeId) + "=" + endpoint + "\n";
        if (nodeId >= 4095) {
            text += "localKvNsId." + std::to_string(nodeId) + "=" + std::to_string(nodeId) + "\n";
        }
    }
    decoded = CreateViewUniverse();
    ASSERT_TRUE(M800ViewCodec::Decode(text, *decoded).ok());
    ASSERT_TRUE(decoded->GetView(1, view).ok());
    EXPECT_EQ(view.asuMap.size(), 4096);
    EXPECT_EQ(view.asuMap.count(4096), 1);
    EXPECT_EQ(view.asuMap.at(4096).localKvNsId, 4096);
    EXPECT_EQ(view.asuMap.count(4097), 0);
}

// 测试场景：编码含 4097 个 ASU 且每个 ASU 都带 localKvNsId 的视图，再解码编码结果。
// 预期结果：仅编码并保留前 4096 个 ASU；ASU 4096 的 localKvNsId 为 4096，ASU 4097 的 asuInfo 和
// localKvNsId 均不输出。
TEST(M800ViewCodecTest, EncodeTruncatesLocalKvNsIdTogetherWithItsAsu)
{
    auto view = MakeView(1, 99);
    for (NodeId nodeId = 1; nodeId <= 4097; ++nodeId) {
        NodeInfo info{{MakeEndpoint()}};
        info.localKvNsId = static_cast<KvNsId>(nodeId);
        view.asuMap.emplace(nodeId, std::move(info));
    }
    auto universe = CreateViewUniverse(
        {
            {1, std::move(view)}
    },
        99);
    std::string encoded;
    ASSERT_TRUE(M800ViewCodec::Encode(*universe, encoded).ok());
    EXPECT_NE(encoded.find("localKvNsId.4096=4096\n"), std::string::npos);
    EXPECT_EQ(encoded.find("asuInfo.4097="), std::string::npos);
    EXPECT_EQ(encoded.find("localKvNsId.4097="), std::string::npos);

    auto decoded = CreateViewUniverse();
    ASSERT_TRUE(M800ViewCodec::Decode(encoded, *decoded).ok());
    GlobalView actual;
    ASSERT_TRUE(decoded->GetView(1, actual).ok());
    EXPECT_EQ(actual.asuMap.size(), 4096);
    EXPECT_EQ(actual.asuMap.at(4096).localKvNsId, 4096);
}

}  // namespace
}  // namespace kv::test
