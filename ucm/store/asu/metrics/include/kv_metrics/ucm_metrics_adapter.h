#pragma once

#include <memory>
#include <string>
#include "kv_metrics/metrics.h"

namespace kv::metrics {

// Writes to the collector registered by Python. Does not register or drain metrics.
std::shared_ptr<KvMetricsBackend> CreateUcmMetricsBackend();

// Install before starting a KV client; all AsuStore instances share the backend.
// Installation failures are logged and not permanently cached. Existing backends are
// not replaced. After all writers stop and Shutdown completes, Setup may reinstall
// the retained backend. Switching backend types with live cached bindings remains
// unsupported; incompatible bindings are skipped with a diagnostic.
// An existing backend is retained and reported at INFO level; the return value
// remains false when InstallBackend rejects a repeated installation.
bool EnsureUcmKvMetricsInstalled(std::string* error = nullptr) noexcept;

}  // namespace kv::metrics
