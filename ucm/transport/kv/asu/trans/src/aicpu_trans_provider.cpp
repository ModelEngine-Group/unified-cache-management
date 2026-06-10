#include "aicpu_trans_provider.h"
#include <atomic>

namespace UC::ASU {
namespace {

std::atomic<AICPUTransProviderSendHook> g_sendHook{nullptr};

}  // namespace

void SetAICPUTransProviderSendHook(AICPUTransProviderSendHook hook)
{
    // Temporary hook for the kv-test fake_backend phase. Production AICPU sends keep the default
    // provider behavior when no hook is registered.
    g_sendHook.store(hook, std::memory_order_release);
}

AICPUTransProviderSendHook GetAICPUTransProviderSendHook()
{
    return g_sendHook.load(std::memory_order_acquire);
}

}  // namespace UC::ASU
