#include "trans_provider.h"
#include <exception>
#include <memory>
#ifdef UCM_ASU_ENABLE_AICPU_PROVIDER
#include "trans_provider/aicpu/aicpu_trans_provider.h"
#endif
#ifdef UCM_ASU_ENABLE_AIV_PROVIDER
#include "trans_provider/aiv/aiv_trans_provider.h"
#endif
#include "kv_transport.h"
#ifdef UCM_ASU_ENABLE_FAKE_PROVIDER
#include "trans_provider/fake/fake_trans_provider.h"
#endif

namespace kv {

Status CreateTransProvider(const TransportConfig& config,
                           std::shared_ptr<TransProvider>& transProvider)
{
    transProvider.reset();
    switch (config.providerType) {
        case TransProviderType::AICPU:
#ifdef UCM_ASU_ENABLE_AICPU_PROVIDER
            transProvider = std::make_shared<AICPUTransProvider>();
            return Status::OK();
#else
            return Status::Error(
                StatusCode::UNSUPPORTED,
                "AICPU trans provider is not built; enable BUILD_UCM_ASU_PROVIDER_AICPU");
#endif
        case TransProviderType::FAKE:
#ifdef UCM_ASU_ENABLE_FAKE_PROVIDER
            try {
                transProvider =
                    std::make_shared<FakeTransProvider>(MakeFakeTransProviderConfig(config));
                return Status::OK();
            } catch (const std::exception& error) {
                return Status::Error(
                    StatusCode::INTERNAL_ERROR,
                    "failed to create fake provider: " + std::string(error.what()));
            }
#else
            return Status::Error(
                StatusCode::UNSUPPORTED,
                "FAKE trans provider is not built; enable BUILD_UCM_ASU_PROVIDER_FAKE");
#endif
        case TransProviderType::AIV:
#ifdef UCM_ASU_ENABLE_AIV_PROVIDER
            transProvider = std::make_shared<AIVTransProviderAdapter>(config.deviceId);
            return Status::OK();
#else
            return Status::Error(
                StatusCode::UNSUPPORTED,
                "AIV trans provider is not built; enable BUILD_UCM_ASU_PROVIDER_AIV and set "
                "ASU_AIV_PROVIDER_ROOT");
#endif
        case TransProviderType::UNSUPPORTED:
            return Status::Error(StatusCode::UNSUPPORTED,
                                 "ASU trans provider backend is not supported");
    }
    return Status::Error(StatusCode::UNSUPPORTED, "ASU trans provider backend is not supported");
}

}  // namespace kv
