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

#include <algorithm>
#include <charconv>
#include <limits>
#include <map>
#include <set>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include "logger.h"
#include "view/m800_view_common.h"

namespace kv {
namespace {

constexpr std::size_t kMaxNamespaces = 64;
constexpr std::size_t kMaxAsusPerNamespace = 4096;
constexpr std::size_t kMaxEndpointsPerAsu = 4;

Status ProtocolError(const std::string& message)
{
    return Status::Error(StatusCode::INTERNAL_ERROR, message);
}

template <typename T>
Status ParseUnsigned(std::string_view text, T& value)
{
    static_assert(std::is_unsigned<T>::value, "T must be unsigned");
    if (text.empty()) { return ProtocolError("empty unsigned integer"); }
    unsigned long long parsed{0};
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, parsed, 10);
    if (result.ec != std::errc{} || result.ptr != end ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
        return ProtocolError("invalid unsigned integer: " + std::string(text));
    }
    value = static_cast<T>(parsed);
    return Status::OK();
}

Status ParseSigned32(std::string_view text, std::int32_t& value)
{
    if (text.empty() || text.front() == '+' || text.front() == '-') {
        return ProtocolError("invalid nonnegative numa_node");
    }
    std::int64_t parsed{0};
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, parsed, 10);
    if (result.ec != std::errc{} || result.ptr != end ||
        parsed < std::numeric_limits<std::int32_t>::min() ||
        parsed > std::numeric_limits<std::int32_t>::max()) {
        return ProtocolError("invalid int32 value: " + std::string(text));
    }
    value = static_cast<std::int32_t>(parsed);
    return Status::OK();
}

bool ContainsForbiddenScalarChar(std::string_view value)
{
    constexpr char forbidden[] = {'\r', '\n', '\0', '=', ',', ';'};
    return value.find_first_of(forbidden, 0, sizeof(forbidden)) != std::string_view::npos;
}

Status ValidateScalar(std::string_view value, bool allowEmpty = false)
{
    if ((!allowEmpty && value.empty()) || ContainsForbiddenScalarChar(value)) {
        return ProtocolError("invalid scalar value in M800 view protocol");
    }
    return Status::OK();
}

std::vector<std::string_view> SplitPreserveEmpty(std::string_view value, char delimiter)
{
    std::vector<std::string_view> parts;
    std::size_t begin{0};
    while (true) {
        const auto pos = value.find(delimiter, begin);
        if (pos == std::string_view::npos) {
            parts.emplace_back(value.substr(begin));
            break;
        }
        parts.emplace_back(value.substr(begin, pos - begin));
        begin = pos + 1;
    }
    return parts;
}

std::string ProtocolName(Protocol protocol)
{
    switch (protocol) {
        case Protocol::UB: return "uboe";
        case Protocol::ROCE: return "roce";
        case Protocol::TCP: return "tcp";
    }
    return {};
}

Status ParseProtocol(std::string_view value, Protocol& protocol)
{
    if (value == "uboe") {
        protocol = Protocol::UB;
    } else if (value == "roce") {
        protocol = Protocol::ROCE;
    } else if (value == "tcp") {
        protocol = Protocol::TCP;
    } else {
        return ProtocolError("invalid endpoint protocol: " + std::string(value));
    }
    return Status::OK();
}

Status ParseEndpoint(std::string_view text, NodeEndpoint& endpoint)
{
    bool hasProtocol{false};
    bool hasPort{false};
    bool hasIp{false};
    bool hasNuma{false};
    bool hasHcaName{false};
    bool hasHcaPort{false};
    std::set<std::string> attrKeys;

    for (const auto field : SplitPreserveEmpty(text, ',')) {
        const auto equals = field.find('=');
        if (equals == std::string_view::npos || equals == 0 ||
            field.find('=', equals + 1) != std::string_view::npos) {
            return ProtocolError("invalid endpoint field");
        }
        const auto key = field.substr(0, equals);
        const auto value = field.substr(equals + 1);
        auto status = ValidateScalar(value);
        if (!status.ok()) { return status; }

        if (key == "protocol") {
            if (hasProtocol) { return ProtocolError("duplicate endpoint protocol"); }
            hasProtocol = true;
            status = ParseProtocol(value, endpoint.protocol);
            if (!status.ok()) { return status; }
            endpoint.attrs["protocol"] = std::string(value);
        } else if (key == "port") {
            if (hasPort) { return ProtocolError("duplicate endpoint port"); }
            hasPort = true;
            status = ParseUnsigned(value, endpoint.port);
            if (!status.ok()) { return status; }
        } else if (key == "local.comm_id") {
            if (hasIp) { return ProtocolError("duplicate endpoint local.comm_id"); }
            hasIp = true;
            endpoint.ip = std::string(value);
        } else if (key == "numa_node") {
            if (hasNuma) { return ProtocolError("duplicate endpoint numa_node"); }
            hasNuma = true;
            status = ParseSigned32(value, endpoint.numaNode);
            if (!status.ok()) { return status; }
        } else if (key == "hca_name") {
            if (hasHcaName) { return ProtocolError("duplicate endpoint hca_name"); }
            hasHcaName = true;
            endpoint.hcaName = std::string(value);
        } else if (key == "hca_port") {
            if (hasHcaPort) { return ProtocolError("duplicate endpoint hca_port"); }
            hasHcaPort = true;
            status = ParseUnsigned(value, endpoint.hcaPort);
            if (!status.ok()) { return status; }
        } else {
            const std::string attrKey{key};
            if (!attrKeys.emplace(attrKey).second) {
                return ProtocolError("duplicate endpoint attribute: " + attrKey);
            }
            endpoint.attrs[attrKey] = std::string(value);
        }
    }

    if (!hasProtocol || !hasPort || !hasIp) {
        return ProtocolError("endpoint requires protocol, port and local.comm_id");
    }
    return Status::OK();
}

Status ParseNodeInfo(std::string_view text, NodeInfo& info)
{
    const auto endpointTexts = SplitPreserveEmpty(text, ';');
    for (const auto endpointText : endpointTexts) {
        NodeEndpoint endpoint;
        auto status = ParseEndpoint(endpointText, endpoint);
        if (!status.ok()) { return status; }
        info.endpoints.emplace_back(std::move(endpoint));
    }
    if (info.endpoints.size() > kMaxEndpointsPerAsu) {
        KV_WARN("M800 ASU has {} endpoints, truncating to {}", info.endpoints.size(),
                kMaxEndpointsPerAsu);
        info.endpoints.resize(kMaxEndpointsPerAsu);
    }
    return Status::OK();
}

Status EncodeEndpoint(const NodeEndpoint& endpoint, std::string& text)
{
    auto status = ValidateScalar(endpoint.ip);
    if (!status.ok()) { return status; }
    const auto protocol = ProtocolName(endpoint.protocol);
    if (protocol.empty()) { return ProtocolError("invalid endpoint protocol enum"); }

    text += "protocol=";
    text += protocol;
    text += ",port=";
    text += std::to_string(endpoint.port);
    text += ",local.comm_id=";
    text += endpoint.ip;
    if (endpoint.numaNode < -1) { return ProtocolError("invalid negative numa_node"); }
    if (endpoint.numaNode != -1) {
        text += ",numa_node=";
        text += std::to_string(endpoint.numaNode);
    }
    if (!endpoint.hcaName.empty()) {
        status = ValidateScalar(endpoint.hcaName);
        if (!status.ok()) { return status; }
        text += ",hca_name=";
        text += endpoint.hcaName;
    }
    if (endpoint.hcaPort != 1) {
        text += ",hca_port=";
        text += std::to_string(endpoint.hcaPort);
    }

    const std::set<std::string> standardKeys = {
        "protocol", "port", "local.comm_id", "numa_node", "hca_name", "hca_port"};
    const std::map<std::string, std::string> orderedAttrs(endpoint.attrs.begin(), endpoint.attrs.end());
    for (const auto& [key, value] : orderedAttrs) {
        if (standardKeys.find(key) != standardKeys.end()) { continue; }
        status = ValidateScalar(key);
        if (!status.ok()) { return status; }
        status = ValidateScalar(value);
        if (!status.ok()) { return status; }
        text.push_back(',');
        text += key;
        text.push_back('=');
        text += value;
    }
    return Status::OK();
}

struct NamespaceBlock {
    KvNsId nsId{0};
    std::uint64_t viewEpoch{0};
    std::uint64_t viewId{0};
    bool hasViewEpoch{false};
    bool hasViewId{false};
    bool hasAsuIds{false};
    std::set<NodeId> asuIds;
    std::map<NodeId, NodeInfo> asuInfos;
    std::map<NodeId, KvNsId> localKvNsIds;
};

Status FinalizeBlock(NamespaceBlock& block, std::uint64_t globalEpoch,
                     std::unordered_map<KvNsId, GlobalView>& views)
{
    if (!block.hasViewEpoch || !block.hasViewId || !block.hasAsuIds) {
        return ProtocolError("namespace block misses viewEpoch, viewId or asuIds");
    }
    if (block.viewEpoch != globalEpoch) {
        return ProtocolError("namespace viewEpoch differs from globalEpoch");
    }
    if (block.viewId != block.nsId) { return ProtocolError("namespace viewId differs from KvNsId"); }
    if (block.asuIds.size() != block.asuInfos.size()) {
        return ProtocolError("asuIds and asuInfo entries differ");
    }
    for (const auto& [nodeId, unused] : block.localKvNsIds) {
        (void)unused;
        if (block.asuIds.find(nodeId) == block.asuIds.end()) {
            return ProtocolError("localKvNsId nodeId is not listed in asuIds");
        }
    }

    GlobalView view;
    view.viewEpoch = block.viewEpoch;
    view.viewId = block.viewId;
    for (const auto nodeId : block.asuIds) {
        const auto info = block.asuInfos.find(nodeId);
        if (info == block.asuInfos.end()) { return ProtocolError("missing asuInfo for listed ASU"); }
        if (view.asuMap.size() == kMaxAsusPerNamespace) {
            KV_WARN("M800 namespace {} ASU list exceeds {}, truncating", block.nsId,
                    kMaxAsusPerNamespace);
            break;
        }
        NodeInfo nodeInfo = info->second;
        const auto localKvNsId = block.localKvNsIds.find(nodeId);
        if (localKvNsId != block.localKvNsIds.end()) {
            nodeInfo.localKvNsId = localKvNsId->second;
        }
        view.asuMap.emplace(nodeId, std::move(nodeInfo));
    }
    for (const auto& [nodeId, unused] : block.asuInfos) {
        (void)unused;
        if (block.asuIds.find(nodeId) == block.asuIds.end()) {
            return ProtocolError("asuInfo is not listed in asuIds");
        }
    }
    views.emplace(block.nsId, std::move(view));
    return Status::OK();
}

}  // namespace

Status M800ViewCodec::Encode(const ViewUniverse& universe, std::string& text)
{
    try {
        std::unordered_map<KvNsId, GlobalView> views;
        std::uint64_t globalEpoch{0};
        universe.GetSnapshot(views, globalEpoch);
        std::vector<KvNsId> nsIds;
        nsIds.reserve(views.size());
        for (const auto& [nsId, unused] : views) {
            (void)unused;
            nsIds.emplace_back(nsId);
        }
        std::sort(nsIds.begin(), nsIds.end());
        if (nsIds.size() > kMaxNamespaces) {
            KV_WARN("M800 universe has {} namespaces, truncating to {}", nsIds.size(), kMaxNamespaces);
            nsIds.resize(kMaxNamespaces);
        }

        std::string output = "globalEpoch=" + std::to_string(globalEpoch) + "\nnsCount=" +
                             std::to_string(nsIds.size()) + "\n";
        for (const auto nsId : nsIds) {
            const auto& view = views.at(nsId);
            if (view.viewEpoch != globalEpoch || view.viewId != nsId) {
                return ProtocolError("namespace epoch/viewId violates unified M800 contract");
            }
            output += "ns=" + std::to_string(nsId) + "\n";
            output += "viewEpoch=" + std::to_string(view.viewEpoch) + "\n";
            output += "viewId=" + std::to_string(view.viewId) + "\n";

            std::vector<NodeId> nodeIds;
            nodeIds.reserve(view.asuMap.size());
            for (const auto& [nodeId, unused] : view.asuMap) {
                (void)unused;
                nodeIds.emplace_back(nodeId);
            }
            std::sort(nodeIds.begin(), nodeIds.end());
            if (nodeIds.size() > kMaxAsusPerNamespace) {
                KV_WARN("M800 namespace {} has {} ASUs, truncating to {}", nsId, nodeIds.size(),
                        kMaxAsusPerNamespace);
                nodeIds.resize(kMaxAsusPerNamespace);
            }
            output += "asuIds=";
            for (std::size_t index = 0; index < nodeIds.size(); ++index) {
                if (index != 0) { output.push_back(','); }
                output += std::to_string(nodeIds[index]);
            }
            output.push_back('\n');

            for (const auto nodeId : nodeIds) {
                const auto& info = view.asuMap.at(nodeId);
                if (info.endpoints.empty()) {
                    return ProtocolError("ASU " + std::to_string(nodeId) + " has no endpoint");
                }
                output += "asuInfo." + std::to_string(nodeId) + "=";
                const auto endpointCount = std::min(info.endpoints.size(), kMaxEndpointsPerAsu);
                if (info.endpoints.size() > kMaxEndpointsPerAsu) {
                    KV_WARN("M800 ASU {} has {} endpoints, truncating to {}", nodeId,
                            info.endpoints.size(), kMaxEndpointsPerAsu);
                }
                for (std::size_t index = 0; index < endpointCount; ++index) {
                    if (index != 0) { output.push_back(';'); }
                    auto status = EncodeEndpoint(info.endpoints[index], output);
                    if (!status.ok()) { return status; }
                }
                output.push_back('\n');
                if (info.localKvNsId != 0) {
                    output += M800ViewTextContract::kLocalKvNsIdFieldPrefix;
                    output += std::to_string(nodeId);
                    output.push_back('=');
                    output += std::to_string(info.localKvNsId);
                    output.push_back('\n');
                }
            }
        }
        text = std::move(output);
        return Status::OK();
    } catch (const std::exception& exception) {
        return ProtocolError(std::string("failed to encode M800 view: ") + exception.what());
    } catch (...) {
        return ProtocolError("failed to encode M800 view: unknown exception");
    }
}

Status M800ViewCodec::Decode(const std::string& text, ViewUniverse& universe)
{
    try {
        if (text.find('\0') != std::string::npos) { return ProtocolError("NUL in M800 view text"); }
        std::uint64_t globalEpoch{0};
        std::uint64_t declaredNsCount{0};
        std::unordered_map<KvNsId, GlobalView> views;
        std::set<KvNsId> namespaceIds;

        const auto lines = SplitPreserveEmpty(text, '\n');
        std::size_t lineIndex{0};
        auto nextContentLine = [&](std::string_view& line, bool& eof) {
            eof = false;
            while (lineIndex < lines.size()) {
                const bool hasLfTerminator = lineIndex + 1 < lines.size();
                line = lines[lineIndex++];
                if (hasLfTerminator && !line.empty() && line.back() == '\r') {
                    line.remove_suffix(1);
                }
                if (line.find('\r') != std::string_view::npos) {
                    return ProtocolError("unexpected CR in M800 view text");
                }
                if (!line.empty()) { return Status::OK(); }
            }
            line = {};
            eof = true;
            return Status::OK();
        };
        auto parseField = [](std::string_view line, std::string_view& key, std::string_view& value) {
            const auto equals = line.find('=');
            if (equals == std::string_view::npos) { return false; }
            key = line.substr(0, equals);
            value = line.substr(equals + 1);
            return true;
        };

        std::string_view line;
        std::string_view key;
        std::string_view value;
        bool eof{false};
        auto status = nextContentLine(line, eof);
        if (!status.ok()) { return status; }
        if (eof) { return ProtocolError("empty M800 view text"); }
        if (!parseField(line, key, value) || key != "globalEpoch") {
            return ProtocolError("first field must be globalEpoch");
        }
        status = ParseUnsigned(value, globalEpoch);
        if (!status.ok()) { return status; }

        status = nextContentLine(line, eof);
        if (!status.ok()) { return status; }
        if (eof) { return ProtocolError("missing nsCount"); }
        if (!parseField(line, key, value) || key != "nsCount") {
            return ProtocolError("second field must be nsCount");
        }
        status = ParseUnsigned(value, declaredNsCount);
        if (!status.ok()) { return status; }

        bool inBlock{false};
        NamespaceBlock block;
        while (true) {
            status = nextContentLine(line, eof);
            if (!status.ok()) { return status; }
            if (eof) { break; }
            const bool hasField = parseField(line, key, value);
            if (hasField && (key == "globalEpoch" || key == "nsCount")) {
                return ProtocolError("duplicate globalEpoch/nsCount");
            }
            if (hasField && key == "ns") {
                if (inBlock) {
                    status = FinalizeBlock(block, globalEpoch, views);
                    if (!status.ok()) { return status; }
                }
                KvNsId nsId{0};
                status = ParseUnsigned(value, nsId);
                if (!status.ok()) { return status; }
                if (!namespaceIds.emplace(nsId).second) {
                    return ProtocolError("duplicate namespace: " + std::to_string(nsId));
                }
                block = NamespaceBlock{};
                block.nsId = nsId;
                inBlock = true;
                continue;
            }
            if (!inBlock) { return ProtocolError("namespace field outside a namespace block"); }
            if (!hasField || key.empty()) {
                return ProtocolError("invalid field inside a namespace block");
            }

            if (key == "viewEpoch") {
                if (block.hasViewEpoch) { return ProtocolError("duplicate viewEpoch"); }
                block.hasViewEpoch = true;
                status = ParseUnsigned(value, block.viewEpoch);
                if (!status.ok()) { return status; }
            } else if (key == "viewId") {
                if (block.hasViewId) { return ProtocolError("duplicate viewId"); }
                block.hasViewId = true;
                status = ParseUnsigned(value, block.viewId);
                if (!status.ok()) { return status; }
            } else if (key == "asuIds") {
                if (block.hasAsuIds) { return ProtocolError("duplicate asuIds"); }
                block.hasAsuIds = true;
                if (!value.empty()) {
                    for (const auto idText : SplitPreserveEmpty(value, ',')) {
                        NodeId nodeId{0};
                        status = ParseUnsigned(idText, nodeId);
                        if (!status.ok()) { return status; }
                        if (!block.asuIds.emplace(nodeId).second) {
                            return ProtocolError("duplicate ASU in asuIds");
                        }
                    }
                }
            } else if (key.rfind(M800ViewTextContract::kAsuInfoFieldPrefix, 0) == 0) {
                NodeId nodeId{0};
                status = ParseUnsigned(
                    key.substr(std::string_view{M800ViewTextContract::kAsuInfoFieldPrefix}.size()),
                    nodeId);
                if (!status.ok()) { return status; }
                if (block.asuInfos.find(nodeId) != block.asuInfos.end()) {
                    return ProtocolError("duplicate asuInfo entry");
                }
                NodeInfo info;
                status = ParseNodeInfo(value, info);
                if (!status.ok()) { return status; }
                block.asuInfos.emplace(nodeId, std::move(info));
            } else if (key.rfind(M800ViewTextContract::kLocalKvNsIdFieldPrefix, 0) == 0) {
                NodeId nodeId{0};
                status = ParseUnsigned(
                    key.substr(
                        std::string_view{M800ViewTextContract::kLocalKvNsIdFieldPrefix}.size()),
                    nodeId);
                if (!status.ok()) { return status; }
                KvNsId localKvNsId{0};
                status = ParseUnsigned(value, localKvNsId);
                if (!status.ok()) { return status; }
                if (!block.localKvNsIds.emplace(nodeId, localKvNsId).second) {
                    return ProtocolError("duplicate localKvNsId entry");
                }
            } else {
                status = ValidateScalar(key);
                if (!status.ok()) { return status; }
                status = ValidateScalar(value);
                if (!status.ok()) { return status; }
            }
        }
        if (inBlock) {
            status = FinalizeBlock(block, globalEpoch, views);
            if (!status.ok()) { return status; }
        }
        if (declaredNsCount != views.size()) {
            return ProtocolError("nsCount differs from actual namespace block count");
        }

        if (views.size() > kMaxNamespaces) {
            std::vector<KvNsId> sortedIds;
            sortedIds.reserve(views.size());
            for (const auto& [nsId, unused] : views) {
                (void)unused;
                sortedIds.emplace_back(nsId);
            }
            std::sort(sortedIds.begin(), sortedIds.end());
            KV_WARN("M800 input has {} namespaces, truncating to {}", sortedIds.size(), kMaxNamespaces);
            for (std::size_t index = kMaxNamespaces; index < sortedIds.size(); ++index) {
                views.erase(sortedIds[index]);
            }
        }
        auto decoded = CreateViewUniverse(std::move(views), globalEpoch);
        return universe.ReplaceAll(*decoded);
    } catch (const std::exception& exception) {
        return ProtocolError(std::string("failed to decode M800 view: ") + exception.what());
    } catch (...) {
        return ProtocolError("failed to decode M800 view: unknown exception");
    }
}

}  // namespace kv
