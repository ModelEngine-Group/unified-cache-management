#ifndef C_COMPRESS_H
#define C_COMPRESS_H
#include <cstddef>
#include "compress_lib/huf.h"

size_t c_compress(const void *src, size_t srcSize, void *dst, size_t dstMax, FixedRatio ratio, DataType dataType);
size_t c_decompress(const void *src, size_t srcSize, void *dst, size_t dstMax, DataType dataType);

#endif // C_COMPRESS_H
