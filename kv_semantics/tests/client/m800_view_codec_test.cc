#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <unordered_map>
#include "view/m800_view_codec.h"
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

TEST(M800ViewCodecTest, DefaultZeroLocalKvNsIdPreservesWireFormat)
{
    EXPECT_STREQ(M800ViewTextContract::kLocalKvNsIdFieldPrefix, "localKvNsId.");
    auto view = MakeView(1, 123);
    view.asuMap.emplace(10, NodeInfo{{MakeEndpoint()}});
    auto universe = CreateViewUniverse({{1, std::move(view)}}, 123);

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

TEST(M800ViewCodecTest, NonZeroLocalKvNsIdRoundTripsWithMultipleEndpoints)
{
    auto view = MakeView(1, 123);
    NodeInfo info{{MakeEndpoint("10.0.0.1"), MakeEndpoint("10.0.0.2")}};
    info.localKvNsId = std::numeric_limits<KvNsId>::max();
    view.asuMap.emplace(10, std::move(info));
    auto universe = CreateViewUniverse({{1, std::move(view)}}, 123);

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

TEST(M800ViewCodecTest, LocalKvNsIdEncodingIsOrderedAndAdjacentToItsAsuInfo)
{
    auto view = MakeView(1, 123);
    NodeInfo node20{{MakeEndpoint("10.0.0.20")}};
    node20.localKvNsId = 20;
    NodeInfo node10{{MakeEndpoint("10.0.0.10")}};
    node10.localKvNsId = 10;
    view.asuMap.emplace(20, std::move(node20));
    view.asuMap.emplace(10, std::move(node10));
    auto universe = CreateViewUniverse({{1, std::move(view)}}, 123);

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
    auto baseline = CreateViewUniverse({{7, std::move(baselineView)}}, 55);

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

TEST(M800ViewCodecTest, DecodeRejectsNonAdjacentLocalKvNsId)
{
    auto beforeAsuInfo = ValidText();
    const auto pos = beforeAsuInfo.find("asuInfo.10=");
    ASSERT_NE(pos, std::string::npos);
    beforeAsuInfo.insert(pos, "localKvNsId.10=1\n");
    const auto separated = ValidText() + "futureField=value\nlocalKvNsId.10=1\n";
    auto baseline = CreateViewUniverse();

    for (const auto& input : {beforeAsuInfo, separated}) {
        EXPECT_EQ(M800ViewCodec::Decode(input, *baseline).code, StatusCode::INTERNAL_ERROR) << input;
        EXPECT_TRUE(baseline->Empty());
        EXPECT_EQ(baseline->GetGlobalEpoch(), 0);
    }
}

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

TEST(M800ViewCodecTest, DecodeAcceptsCrlfBlankLinesAndMissingFinalNewline)
{
    std::string text = ValidText();
    text.erase(text.size() - 1);
    for (std::size_t pos = 0; (pos = text.find('\n', pos)) != std::string::npos; pos += 2) {
        text.replace(pos, 1, "\r\n");
    }
    text.insert(0, "\r\n");

    auto decoded = CreateViewUniverse();
    ASSERT_TRUE(M800ViewCodec::Decode(text, *decoded).ok()) << text;
    EXPECT_EQ(decoded->GetAllNsIds().size(), 1);
}

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
        {"nsCount=1", "nsCount=2"},
        {"viewEpoch=123", "viewEpoch=122"},
        {"viewId=1", "viewId=2"},
        {"viewId=1", "viewId=1\nviewId=1"},
        {"asuIds=10", "asuIds=10,11"},
        {"asuInfo.10=", "asuInfo.11="},
        {"port=6000", "port=65536"},
        {"port=6000", "port=6000,numa_node=-1"},
        {"protocol=uboe", "protocol=unknown"},
        {"local.comm_id=10.0.0.1", "local.comm_id="},
        {"asuInfo.10", "asuInfo.10x"},
        {"port=6000", "port=60=00"},
        {"asuIds=10", "this-is-not-a-field\nasuIds=10"},
        {"asuIds=10", "=value\nasuIds=10"},
        {"asuIds=10", "futureField=a=b\nasuIds=10"},
        {"asuIds=10", "futureField=a,b\nasuIds=10"},
        {"ns=1\n", "ns=1\nns=1\n"},
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

    auto baseline = CreateViewUniverse({{7, MakeView(7, 55)}}, 55);
    for (const auto& input : invalidInputs) {
        auto status = M800ViewCodec::Decode(input, *baseline);
        EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR) << input;
        EXPECT_EQ(baseline->GetGlobalEpoch(), 55) << input;
        ASSERT_EQ(baseline->GetAllNsIds().size(), 1) << input;
        EXPECT_EQ(baseline->GetAllNsIds().front(), 7) << input;
    }
}

TEST(M800ViewCodecTest, UnknownBlockFieldIsIgnoredButOutsideFieldIsRejected)
{
    auto input = ValidText();
    input.insert(input.find("asuIds="), "futureField=anything\n");
    auto decoded = CreateViewUniverse();
    EXPECT_TRUE(M800ViewCodec::Decode(input, *decoded).ok());

    input = ValidText();
    input.insert(input.find("ns=1\n"), "futureField=anything\n");
    auto unchanged = CreateViewUniverse({{7, MakeView(7, 55)}}, 55);
    EXPECT_EQ(M800ViewCodec::Decode(input, *unchanged).code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(unchanged->GetGlobalEpoch(), 55);
}

TEST(M800ViewCodecTest, EncodeRejectsInconsistentViewAndInvalidScalar)
{
    auto invalid = MakeView(1, 10);
    invalid.viewEpoch = 9;
    auto universe = CreateViewUniverse({{1, invalid}}, 10);
    std::string output = "unchanged";
    EXPECT_EQ(M800ViewCodec::Encode(*universe, output).code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(output, "unchanged");

    auto valid = MakeView(1, 10);
    auto endpoint = MakeEndpoint();
    endpoint.attrs["bad"] = "a,b";
    valid.asuMap.emplace(1, NodeInfo{{endpoint}});
    universe = CreateViewUniverse({{1, valid}}, 10);
    EXPECT_EQ(M800ViewCodec::Encode(*universe, output).code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(output, "unchanged");
}

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

TEST(M800ViewCodecTest, DecodeDirectlyTruncatesOversizedWireInput)
{
    const std::string endpoint = "protocol=uboe,port=6000,local.comm_id=10.0.0.1";
    std::string text = "globalEpoch=99\nnsCount=65\n";
    for (KvNsId nsId = 1; nsId <= 65; ++nsId) {
        text += "ns=" + std::to_string(nsId) + "\nviewEpoch=99\nviewId=" +
                std::to_string(nsId) + "\nasuIds=1\nasuInfo.1=" + endpoint + "\n";
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

TEST(M800ViewCodecTest, EncodeTruncatesLocalKvNsIdTogetherWithItsAsu)
{
    auto view = MakeView(1, 99);
    for (NodeId nodeId = 1; nodeId <= 4097; ++nodeId) {
        NodeInfo info{{MakeEndpoint()}};
        info.localKvNsId = static_cast<KvNsId>(nodeId);
        view.asuMap.emplace(nodeId, std::move(info));
    }
    auto universe = CreateViewUniverse({{1, std::move(view)}}, 99);
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
