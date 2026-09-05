#include "hcomm_config_adapter.h"

namespace kv::bench {

namespace {

constexpr int kExitInvalidArgument = 1;

}  // namespace

Status HcommConfigAdapter::ResolveProtocol(kv::Protocol protocol, const KvTestConfig& config,
                                           HcommProtocol& hcommProtocol) const
{
    switch (protocol) {
        case kv::Protocol::UB:
            hcommProtocol = config.hcommProtocolMapping.ub;
            return Status::Success();
        case kv::Protocol::ROCE:
            hcommProtocol = config.hcommProtocolMapping.roce;
            return Status::Success();
        case kv::Protocol::TCP:
            // TODO(#6): Confirm whether ASU TCP should map to Hcomm COMM_PROTOCOL_PCIE.
            hcommProtocol = config.hcommProtocolMapping.tcp;
            return Status::Success();
        default: return Status::Error(kExitInvalidArgument, "unsupported ASU protocol for Hcomm");
    }
}

Status HcommConfigAdapter::ValidateChannelSource(const KvTestConfig& config) const
{
    if (config.behavior.hcommLocalRole != HcommLocalRole::SERVER) {
        return Status::Error(kExitInvalidArgument, "kv-test Hcomm local role must be server");
    }
    if (config.behavior.hcommChannelConfigSource != HcommChannelConfigSource::ASU_CONFIG) {
        return Status::Error(kExitInvalidArgument,
                             "kv-test Hcomm channel config source must be ASU config");
    }
    if (config.behavior.hcommApiBoundary != HcommApiBoundary::C_API) {
        return Status::Error(kExitInvalidArgument, "kv-test Hcomm API boundary must be C API");
    }
    if (config.behavior.wireProtocol != WireProtocol::SQE) {
        return Status::Error(kExitInvalidArgument, "kv-test wire protocol must be SQE");
    }
    if (config.behavior.transportLinkPolicy != TransportLinkPolicy::SHARED_DATA_LINK) {
        return Status::Error(kExitInvalidArgument,
                             "kv-test transport link policy must use shared data link");
    }
    if (config.asuClientConfig.transportConfigs.empty()) {
        return Status::Error(kExitInvalidArgument,
                             "ASU transport config is required for Hcomm channel source");
    }

    for (const auto& transportConfig : config.asuClientConfig.transportConfigs) {
        if (transportConfig.endpoints.empty()) {
            return Status::Error(kExitInvalidArgument,
                                 "ASU transport endpoint is required for Hcomm channel source, "
                                 "asuId=" +
                                     std::to_string(transportConfig.asuId));
        }
        for (const auto& endpoint : transportConfig.endpoints) {
            if (endpoint.ip.empty()) {
                return Status::Error(kExitInvalidArgument,
                                     "ASU endpoint local socket/comm_id is required for Hcomm, "
                                     "asuId=" +
                                         std::to_string(transportConfig.asuId));
            }
            if (endpoint.port == 0) {
                return Status::Error(kExitInvalidArgument,
                                     "ASU endpoint port is required for Hcomm, asuId=" +
                                         std::to_string(transportConfig.asuId));
            }

            HcommProtocol hcommProtocol{HcommProtocol::ROCE};
            auto status = ResolveProtocol(endpoint.protocol, config, hcommProtocol);
            if (!status.Ok()) { return status; }
        }
    }

    return Status::Success();
}

}  // namespace kv::bench
