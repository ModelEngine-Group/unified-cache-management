#ifndef UNIFIEDCACHE_STORE_CC_COMPRESSOR_H
#define UNIFIEDCACHE_STORE_CC_COMPRESSOR_H

#include <memory>
#include "global_config.h"
#include "ucmstore_v1.h"

namespace UC::Compressor {

class CompressorImpl;
class Compressor : public StoreV1 {
public:
    ~Compressor() override;
    Status Setup(const Config& config);
    std::string Readme() const override;
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override;
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override;
    void Prefetch(const Detail::BlockId* blocks, size_t num) override;
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override;
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override;
    Expected<bool> Check(Detail::TaskHandle taskId) override;
    Status Wait(Detail::TaskHandle taskId) override;

private:
    std::shared_ptr<CompressorImpl> impl_;
};

} // namespace UC::Compressor

#endif
