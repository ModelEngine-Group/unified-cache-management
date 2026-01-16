#include <cstring>
#include <cstdlib>
#include "c_compress.h"


// 逐个shard压缩 src->dst
// 返回压缩后的字节数
size_t c_compress(const void *src, size_t srcSize, void *dst, size_t dstMax, FixedRatio ratio, DataType dataType) {
    return HUF_compress_float_fixRatio (dst, dstMax, src, srcSize, ratio, dataType);
}

// 逐个shard解压 src->dst
// 返回解压后的字节数
size_t c_decompress(const void *src, size_t srcSize, void *dst, size_t dstMax, DataType dataType) {
    return HUF_decompress_float_fixRatio (dst, dstMax, src, srcSize, dataType);
}
 

