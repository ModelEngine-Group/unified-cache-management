#include <nds_interface.h>
#include <cstdint>
#include <cstddef>
#include <vector>

extern "C" int32_t c_init(void *memAddr, uint64_t length, const char *path_nds_config)
{
    return NDS::init(memAddr, length, path_nds_config);
}

extern "C" int32_t c_init_multi(const void *const *memAddrs, const uint64_t *lengths, size_t count, const char *path_nds_config)
{
    if (count > 0 && (memAddrs == nullptr || lengths == nullptr)) {
        return -1;
    }
    // SDK 注册 MR 不会经指针回写，const_cast 仅为适配 NDS::initMulti 的 void* 形参
    std::vector<void*> addrVec;
    std::vector<uint64_t> lenVec;
    addrVec.reserve(count);
    lenVec.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        addrVec.push_back(const_cast<void*>(memAddrs[i]));
        lenVec.push_back(lengths[i]);
    }
    return NDS::initMulti(addrVec, lenVec, path_nds_config);
}

extern "C" int32_t c_isExists(const uint64_t *blockIds, size_t count, uint32_t nsid)
{
    // C API 仍为 64-bit key（key_high 恒 0）；80-bit key 请走 C++ / Python 接口。
    // 注意 wrapper 默认 xxhash 打散对本路径同样生效。
    std::vector<uint64_t> vec(blockIds, blockIds + count);
    std::vector<uint16_t> highs(count, 0);
    return NDS::isExists(vec, highs, nsid);
}

extern "C" int32_t c_get(uint64_t blockId, uint8_t *blockAddr, size_t offset, size_t length, uint32_t nsid)
{
    return NDS::get(blockId, 0, blockAddr, offset, length, nsid);
}

extern "C" int32_t c_batchGet(const uint64_t *blockIds, uint8_t **blockAddrs,
                              const size_t *offsets, const size_t *lengths, const uint32_t *nsids, size_t count)
{
    std::vector<uint64_t> idVec(blockIds, blockIds + count);
    std::vector<uint16_t> highVec(count, 0);
    std::vector<uint8_t *> addrVec(blockAddrs, blockAddrs + count);
    std::vector<size_t> offVec(offsets, offsets + count);
    std::vector<size_t> lenVec(lengths, lengths + count);
    std::vector<uint32_t> nsidVec(nsids, nsids + count);
    return NDS::batchGet(idVec, highVec, addrVec, offVec, lenVec, nsidVec);
}

extern "C" int32_t c_put(uint64_t blockId, uint8_t *blockAddr, size_t offset, size_t length, uint32_t nsid)
{
    return NDS::put(blockId, 0, blockAddr, offset, length, nsid);
}

extern "C" int32_t c_batchPut(const uint64_t *blockIds, uint8_t **blockAddrs,
                              const size_t *offsets, const size_t *lengths, const uint32_t *nsids, size_t count)
{
    std::vector<uint64_t> idVec(blockIds, blockIds + count);
    std::vector<uint16_t> highVec(count, 0);
    std::vector<uint8_t *> addrVec(blockAddrs, blockAddrs + count);
    std::vector<size_t> offVec(offsets, offsets + count);
    std::vector<size_t> lenVec(lengths, lengths + count);
    std::vector<uint32_t> nsidVec(nsids, nsids + count);
    return NDS::batchPut(idVec, highVec, addrVec, offVec, lenVec, nsidVec);
}