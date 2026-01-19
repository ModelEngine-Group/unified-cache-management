/* ******************************************************************
 * hist : Histogram functions
 * part of Finite State Entropy project
 * Copyright (c) 2013-2020, Yann Collet, Facebook, Inc.
 *
 *  You can contact the author at :
 *  - FSE source repository : https://github.com/Cyan4973/FiniteStateEntropy 
 *  - Public forum : https://groups.google.com/forum/# !forum/lz4c
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
****************************************************************** */

/* --- dependencies --- */
#include "mem.h"             /* U32, BYTE, etc. */
#include "debug.h"           /* assert, DEBUGLOG */
#include "error_private.h"   /* ERROR */
#include "hist.h"


/* --- Error management --- */
unsigned HIST_isError(size_t code) { return ERR_isError(code); }

/*-**************************************************************
 *  Histogram functions
 ****************************************************************/
unsigned HIST_count_simple(unsigned* count, unsigned* maxSymbolValuePtr,
                           const void* src, size_t srcSize)
{
    const BYTE* ip = (const BYTE*)src;
    const BYTE* const end = ip + srcSize;
    unsigned maxSymbolValue = *maxSymbolValuePtr;
    unsigned largestCount=0;

    memset(count, 0, (maxSymbolValue+1) * sizeof(*count));
    if (srcSize==0) { *maxSymbolValuePtr = 0; return 0; }

    while (ip<end) {
        assert(*ip <= maxSymbolValue);
        count[*ip++]++;
    }

    while (!count[maxSymbolValue]) maxSymbolValue--;
    *maxSymbolValuePtr = maxSymbolValue;

    {   U32 s;
        for (s=0; s<=maxSymbolValue; s++)
            if (count[s] > largestCount) largestCount = count[s];
    }

    return largestCount;
}

typedef enum { trustInput, checkMaxSymbolValue } HIST_checkInput_e;

/* HIST_count_parallel_wksp() :
 * store histogram into 4 intermediate tables, recombined at the end.
 * this design makes better use of OoO cpus,
 * and is noticeably faster when some values are heavily repeated.
 * But it needs some additional workspace for intermediate tables.
 * `workSpace` must be a U32 table of size >= HIST_WKSP_SIZE_U32.
 * @return : largest histogram frequency,
 *           or an error code (notably when histogram's alphabet is larger than *maxSymbolValuePtr) */
static size_t HIST_count_parallel_wksp(
                                unsigned* count, unsigned* maxSymbolValuePtr,
                                const void* source, size_t sourceSize,
                                HIST_checkInput_e check,
                                U32* const workSpace)
{
    const BYTE* ip = (const BYTE*)source;
    const BYTE* const iend = ip+sourceSize;
    size_t const countSize = (*maxSymbolValuePtr + 1) * sizeof(*count);
    unsigned max=0;
    U32* const Counting1 = workSpace;
    U32* const Counting2 = Counting1 + 256;
    U32* const Counting3 = Counting2 + 256;
    U32* const Counting4 = Counting3 + 256;

    /* safety checks */
    assert(*maxSymbolValuePtr <= 255);
    if (!sourceSize) {
        memset(count, 0, countSize);
        *maxSymbolValuePtr = 0;
        return 0;
    }
    memset(workSpace, 0, 4*256*sizeof(unsigned));

    /* by stripes of 16 bytes */
    {   U32 cached = MEM_read32(ip); ip += 4;
        while (ip < iend-15) {
            U32 c = cached; cached = MEM_read32(ip); ip += 4;
            Counting1[(BYTE) c     ]++;
            Counting2[(BYTE)(c>>8) ]++;
            Counting3[(BYTE)(c>>16)]++;
            Counting4[       c>>24 ]++;
            c = cached; cached = MEM_read32(ip); ip += 4;
            Counting1[(BYTE) c     ]++;
            Counting2[(BYTE)(c>>8) ]++;
            Counting3[(BYTE)(c>>16)]++;
            Counting4[       c>>24 ]++;
            c = cached; cached = MEM_read32(ip); ip += 4;
            Counting1[(BYTE) c     ]++;
            Counting2[(BYTE)(c>>8) ]++;
            Counting3[(BYTE)(c>>16)]++;
            Counting4[       c>>24 ]++;
            c = cached; cached = MEM_read32(ip); ip += 4;
            Counting1[(BYTE) c     ]++;
            Counting2[(BYTE)(c>>8) ]++;
            Counting3[(BYTE)(c>>16)]++;
            Counting4[       c>>24 ]++;
        }
        ip-=4;
    }

    /* finish last symbols */
    while (ip<iend) Counting1[*ip++]++;

    {   U32 s;
        for (s=0; s<256; s++) {
            Counting1[s] += Counting2[s] + Counting3[s] + Counting4[s];
            if (Counting1[s] > max) max = Counting1[s];
    }   }

    {   unsigned maxSymbolValue = 255;
        while (!Counting1[maxSymbolValue]) maxSymbolValue--;
        if (check && maxSymbolValue > *maxSymbolValuePtr) return ERROR(maxSymbolValue_tooSmall);
        *maxSymbolValuePtr = maxSymbolValue;
        memmove(count, Counting1, countSize);   /* in case count & Counting1 are overlapping */
    }
    return (size_t)max;
}

size_t HIST_count_BF16(unsigned* count, unsigned* maxSymbolValuePtr,
                       const void* src, size_t srcSize, 
                       void *dst, size_t dstCapacity, size_t* compressionSize)
{
    const uint16_t* ip = (const uint16_t*)src;
    const uint16_t* const iend = ip + srcSize;
    size_t const countSize = (*maxSymbolValuePtr + 1) * sizeof(*count);
    unsigned largestCount = 0;
    BYTE* const ostart = (BYTE*) dst;
    BYTE* op = ostart;
    BYTE* oend = (BYTE*)dst + dstCapacity;

    /* safety checks */
    assert(*maxSymbolValuePtr <= 255);
    if (!srcSize) {
        memset(count, 0, countSize);
        *maxSymbolValuePtr = 0;
        return 0;
    }

    /* counting */
    while (ip < iend && op < oend) {
        // exponent_buffer[i] = (buffer[i] >> 7) & 0xFF;  // 提取指数位
        count[((*ip)>>7) & 0xFF] ++;
        *op++ = (((*ip >> 15) & 0x1) << 7) | (*ip & 0x7F);
        ip ++;
    }
    
    *compressionSize = op - ostart;
    {   U32 s;
        for (s=0; s<256; s++) {
            if (count[s] > largestCount) largestCount = count[s];
    }   }

    {   unsigned maxSymbolValue = 255;
        while (!count[maxSymbolValue]) maxSymbolValue--;
        if (maxSymbolValue > *maxSymbolValuePtr) return ERROR(maxSymbolValue_tooSmall);
        *maxSymbolValuePtr = maxSymbolValue;
    }
    return (size_t)largestCount;
}


size_t HIST_count_BF16_fixRatio(unsigned* count, unsigned* maxSymbolValuePtr, const void* src, size_t srcSize) {
    if (srcSize == 0) {
        return 0;
    }
    
    // safety checks
    assert(*maxSymbolValuePtr <= 255);

    // 统计直方图
    const uint16_t* ip = (const uint16_t*)src;
    for (size_t i=0; i<srcSize; i++) {
        count[ ((ip[i])>>7) & 0xFF ] ++;  // 提取指数位
    }
    
    // 找出统计值最大的symbol
    U32 count_largest = 0;
    U32 s_largest = 0;
    {   U32 s;
        for (s=0; s<256; s++) {
            if (count_largest < count[s]) {
                count_largest = count[s];
                s_largest = s;
            }
        }
    }

    // 如果只有一个symbol，特殊处理一下，避免采用RLE编码
    if (count_largest == srcSize) {
        if (s_largest == 0) {
            count[1]           = 1;
        } else {
            count[s_largest-1] = 1;
        }
    }

    {   unsigned maxSymbolValue = 255;
        while (!count[maxSymbolValue]) maxSymbolValue--;
        if (maxSymbolValue > *maxSymbolValuePtr) return ERROR(maxSymbolValue_tooSmall);
        *maxSymbolValuePtr = maxSymbolValue;
    }

    return (size_t)count_largest;
}


size_t HIST_count_FP16(unsigned* count, unsigned* maxSymbolValuePtr,
                       const void* src, size_t srcSize, 
                       void *dst, size_t dstCapacity, size_t* compressionSize)
{
    const uint16_t* ip = (const uint16_t*)src;
    const uint16_t* const iend = ip + srcSize;
    size_t const countSize = (*maxSymbolValuePtr + 1) * sizeof(*count);
    unsigned largestCount = 0;
    BYTE* const ostart = (BYTE*) dst;
    BYTE* op = ostart;
    BYTE* oend = (BYTE*)dst + dstCapacity;

    /* safety checks */
    assert(*maxSymbolValuePtr <= 255);
    if (!srcSize) {
        memset(count, 0, countSize);
        *maxSymbolValuePtr = 0;
        return 0;
    }

    // 一次处理 64 个数 8*8 
    // | 8B s | 16B m2 | 64B m8 | 8B s | 16B m2 | 64B m8 | 8B s | 16B m2 | 64B m8 |
    uint8_t* sign = op;
    uint16_t* mantissa2 = (uint16_t*)(op + 8);
    uint64_t* mantissa8 = (uint64_t*)(op + 24);

    assert(srcSize%64 == 0);
    while (ip + 64 <= iend && op + 88 <= oend) {

        for (int g = 0; g < 8; g++) {   // 每组 8 个数
            // 一次读 8 个字节（两个 U32）
            U64 c1 = MEM_read64(ip); ip += 4;
            U64 c2 = MEM_read64(ip); ip += 4;

            // 展开 8 次，每次处理 1 个字节
            uint16_t vals[8] = {(uint16_t)(c1), (uint16_t)(c1>>16), (uint16_t)(c1>>32), (uint16_t)(c1>>48),
                                (uint16_t)(c2), (uint16_t)(c2>>16), (uint16_t)(c2>>32), (uint16_t)(c2>>48)};

            for (int pos=0; pos<8; pos++) {
                uint16_t v = vals[pos];
                unsigned sym = (v >> 10) & 0x1F;
                count[sym]++;

                sign[pos] |= ((v >> 15) & 0x1) << g;
                mantissa2[pos] |= (uint16_t)((v >> 8) & 0x3) << (g*2);
                mantissa8[pos] |= (uint64_t)(v & 0xFF) << (g*8);
            }
        }

        op += 88;
        sign = op;
        mantissa2 = (uint16_t*)(op + 8);
        mantissa8 = (uint64_t*)(op + 24);
    }
    
    *compressionSize = op - ostart;
    {   U32 s;
        for (s=0; s<256; s++) {
            if (count[s] > largestCount) largestCount = count[s];
    }   }

    {   unsigned maxSymbolValue = 255;
        while (!count[maxSymbolValue]) maxSymbolValue--;
        if (maxSymbolValue > *maxSymbolValuePtr) return ERROR(maxSymbolValue_tooSmall);
        *maxSymbolValuePtr = maxSymbolValue;
    }
    return (size_t)largestCount;
}

size_t HIST_lossy_count_FP16(unsigned* count, unsigned* maxSymbolValuePtr,
                             const void* src, size_t srcSize, 
                             void *dst, size_t dstCapacity, size_t* compressionSize)
{
    const uint16_t* ip = (const uint16_t*)src;
    const uint16_t* const iend = ip + srcSize;
    size_t const countSize = (*maxSymbolValuePtr + 1) * sizeof(*count);
    unsigned largestCount = 0;
    BYTE* const ostart = (BYTE*) dst;
    BYTE* op = ostart;
    BYTE* oend = (BYTE*)dst + dstCapacity;

    /* safety checks */
    assert(*maxSymbolValuePtr <= 255);
    if (!srcSize) {
        memset(count, 0, countSize);
        *maxSymbolValuePtr = 0;
        return 0;
    }

    // 一次处理 64 个数 8*8 , 尾数部分截断保留4位
    // | 8B s | 32B m4 | 8B s | 32B m4| 8B s | 32B m4 | 8B s | 32B m4 |
    uint8_t* sign = op;
    uint32_t* mantissa4 = (uint32_t*)(op + 8);

    assert(srcSize%64 == 0);
    while (ip + 64 <= iend && op + 40 <= oend) {

        for (int g = 0; g < 8; g++) {   // 每组 8 个数
            // 一次读 8 个字节（两个 U32）
            U64 c1 = MEM_read64(ip); ip += 4;
            U64 c2 = MEM_read64(ip); ip += 4;

            // 展开 8 次，每次处理 1 个字节
            uint16_t vals[8] = {(uint16_t)(c1), (uint16_t)(c1>>16), (uint16_t)(c1>>32), (uint16_t)(c1>>48),
                                (uint16_t)(c2), (uint16_t)(c2>>16), (uint16_t)(c2>>32), (uint16_t)(c2>>48)};

            for (int pos=0; pos<8; pos++) {
                uint16_t v = vals[pos];
                unsigned sym = (v >> 10) & 0x1F;
                count[sym]++;

                sign[pos] |= ((v >> 15) & 0x1) << g;
                mantissa4[pos] |= (uint16_t)((v >> 6) & 0xF) << (g*4);
            }
        }

        op += 40;
        sign = op;
        mantissa4 = (uint32_t*)(op + 8);
    }
    
    *compressionSize = op - ostart;
    {   U32 s;
        for (s=0; s<256; s++) {
            if (count[s] > largestCount) largestCount = count[s];
    }   }

    {   unsigned maxSymbolValue = 255;
        while (!count[maxSymbolValue]) maxSymbolValue--;
        if (maxSymbolValue > *maxSymbolValuePtr) return ERROR(maxSymbolValue_tooSmall);
        *maxSymbolValuePtr = maxSymbolValue;
    }
    return (size_t)largestCount;
}

size_t HIST_count_FP8E5M2(unsigned* count, unsigned* maxSymbolValuePtr,
                       const void* src, size_t srcSize, 
                       void *dst, size_t dstCapacity, size_t* compressionSize)
{
    const uint8_t* ip = (const uint8_t*)src;
    const uint8_t* const iend = ip + srcSize;
    size_t const countSize = (*maxSymbolValuePtr + 1) * sizeof(*count);
    unsigned largestCount = 0;
    BYTE* const ostart = (BYTE*) dst;
    BYTE* op = ostart;
    BYTE* oend = (BYTE*)dst + dstCapacity;

    /* safety checks */
    assert(*maxSymbolValuePtr <= 255);
    if (!srcSize) {
        memset(count, 0, countSize);
        *maxSymbolValuePtr = 0;
        return 0;
    }

    // 一次处理 64 个数 8*8 
    // | 8B s | 16B m | 8B s | 16B m | 8B s | 16B m |
    uint8_t* sign = op;
    uint16_t* mantissa16 = (uint16_t*)(op + 8);

    assert(srcSize%64 == 0);
    while (ip + 64 <= iend && op + 24 <= oend) {

        for (int g = 0; g < 8; g++) {   // 每组 8 个数
            // 一次读 8 个字节（两个 U32）
            U32 c1 = MEM_read32(ip); ip += 4;
            U32 c2 = MEM_read32(ip); ip += 4;

            // 展开 8 次，每次处理 1 个字节
            uint8_t vals[8] = {(uint8_t)(c1), (uint8_t)(c1>>8), (uint8_t)(c1>>16), (uint8_t)(c1>>24),
                               (uint8_t)(c2), (uint8_t)(c2>>8), (uint8_t)(c2>>16), (uint8_t)(c2>>24)};

            for (int pos=0; pos<8; pos++) {
                uint8_t v = vals[pos];
                unsigned sym = (v >> 2) & 0x1F;
                count[sym]++;

                sign[pos] |= ((v >> 7) & 0x1) << g;
                mantissa16[pos] |= (uint16_t)(v & 0x3) << (g*2);
            }
        }

        op += 24;
        sign = op;
        mantissa16 = (uint16_t*)(op + 8);
    }
    
    *compressionSize = op - ostart;
    {   U32 s;
        for (s=0; s<256; s++) {
            if (count[s] > largestCount) largestCount = count[s];
    }   }

    {   unsigned maxSymbolValue = 255;
        while (!count[maxSymbolValue]) maxSymbolValue--;
        if (maxSymbolValue > *maxSymbolValuePtr) return ERROR(maxSymbolValue_tooSmall);
        *maxSymbolValuePtr = maxSymbolValue;
    }
    return (size_t)largestCount;
}

/* HIST_countFast_wksp() :
 * Same as HIST_countFast(), but using an externally provided scratch buffer.
 * `workSpace` is a writable buffer which must be 4-bytes aligned,
 * `workSpaceSize` must be >= HIST_WKSP_SIZE
 */
size_t HIST_countFast_wksp(unsigned* count, unsigned* maxSymbolValuePtr,
                          const void* source, size_t sourceSize,
                          void* workSpace, size_t workSpaceSize)
{
    if (sourceSize < 1500) /* heuristic threshold */
        return HIST_count_simple(count, maxSymbolValuePtr, source, sourceSize);
    if ((size_t)workSpace & 3) return ERROR(GENERIC);  /* must be aligned on 4-bytes boundaries */
    if (workSpaceSize < HIST_WKSP_SIZE) return ERROR(workSpace_tooSmall);
    return HIST_count_parallel_wksp(count, maxSymbolValuePtr, source, sourceSize, trustInput, (U32*)workSpace);
}

/* fast variant (unsafe : won't check if src contains values beyond count[] limit) */
size_t HIST_countFast(unsigned* count, unsigned* maxSymbolValuePtr,
                     const void* source, size_t sourceSize)
{
    unsigned tmpCounters[HIST_WKSP_SIZE_U32];
    return HIST_countFast_wksp(count, maxSymbolValuePtr, source, sourceSize, tmpCounters, sizeof(tmpCounters));
}

/* HIST_count_wksp() :
 * Same as HIST_count(), but using an externally provided scratch buffer.
 * `workSpace` size must be table of >= HIST_WKSP_SIZE_U32 unsigned */
size_t HIST_count_wksp(unsigned* count, unsigned* maxSymbolValuePtr,
                       const void* source, size_t sourceSize,
                       void* workSpace, size_t workSpaceSize)
{
    if ((size_t)workSpace & 3) return ERROR(GENERIC);  /* must be aligned on 4-bytes boundaries */
    if (workSpaceSize < HIST_WKSP_SIZE) return ERROR(workSpace_tooSmall);
    if (*maxSymbolValuePtr < 255)
        return HIST_count_parallel_wksp(count, maxSymbolValuePtr, source, sourceSize, checkMaxSymbolValue, (U32*)workSpace);
    *maxSymbolValuePtr = 255;
    return HIST_countFast_wksp(count, maxSymbolValuePtr, source, sourceSize, workSpace, workSpaceSize);
}

size_t HIST_count(unsigned* count, unsigned* maxSymbolValuePtr,
                 const void* src, size_t srcSize)
{
    unsigned tmpCounters[HIST_WKSP_SIZE_U32];
    return HIST_count_wksp(count, maxSymbolValuePtr, src, srcSize, tmpCounters, sizeof(tmpCounters));
}
