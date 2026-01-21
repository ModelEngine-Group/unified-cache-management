#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>

#include "../compress_lib/huf.h"


static void set_BF16_e (uint16_t* buf, size_t count) {
    for (size_t i=0; i<count; i++) {
        buf[i] &= 0x807F;
        buf[i] |= 0x4000;
    }
}


static void compare_BF16 (const uint16_t* buf1, const uint16_t* buf2, size_t count) {
    size_t count_same = 0;
    size_t count_half = 0;
    size_t count_error= 0;
    for (size_t i=0; i<count; i++) {
        uint16_t v1 = buf1[i];
        uint16_t v2 = buf2[i];
        if (v1 == v2) {                           // BF16值完全相同
            count_same ++;
        } else if ((v1&0xFFF0) == (v2&0xFFF0)) {  // 只有尾数低4-bit不同
            count_half ++;
        } else {                                  // 错误
            count_error ++;
        }
    }
    printf("总数=%lu,  完全相同=%lu,  仅低4bit不同=%lu,  不同=%lu\n", count, count_same, count_half, count_error);
    if (count_error) {
        printf("*** 测试不通过 ***\n");
    } else {
        printf("=== 测试通过 ===\n");
    }
}

#define CHUNK_SIZE 131072

// 单测主函数
int test_huf_compression(const char* filename) {
    printf("Input file: %s\n", filename);
    
    for (int i = 0; i < 144; ++i) {
        /* 1. 从文件读取BF16 tensor数据 */
        FILE* fp = fopen(filename, "rb");
        if (!fp) {
            printf("Error: Failed to open file '%s'\n", filename);
            return -1;
        }
        
        /* 获取文件大小 */
        fseek(fp, 0, SEEK_END);
        size_t srcSize = CHUNK_SIZE;
        fseek(fp, CHUNK_SIZE*i, SEEK_SET);
        
        /* 验证文件大小是否为2字节的整数倍（BF16要求） */
        if (srcSize % 2 != 0) {
            printf("Error: File size (%zu) is not a multiple of 2 bytes\n", srcSize);
            fclose(fp);
            return -1;
        }
        
        /* 分配buffer存储原始数据 */
        uint8_t* srcBuf = (uint8_t*)malloc(srcSize);
        if (!srcBuf) {
            printf("Error: Failed to allocate memory for source buffer\n");
            fclose(fp);
            return -1;
        }
        
        /* 读取文件数据 */
        size_t readBytes = fread(srcBuf, 1, srcSize, fp);
        if (readBytes != srcSize) {
            printf("Error: Failed to read complete file (expected %zu, got %zu)\n", srcSize, readBytes);
            free(srcBuf);
            return -1;
        }
        
        /* 2. 准备压缩buffer */
        /* 分配足够大的buffer，通常压缩数据不应超过原始数据大小 */
        size_t compBufSize = srcSize;  /* 最坏情况：无法压缩 */
        uint8_t* compBuf = (uint8_t*)malloc(compBufSize);
        if (!compBuf) {
            printf("Error: Failed to allocate memory for compression buffer\n");
            free(srcBuf);
            return -1;
        }

        // 可选：把待压缩数据的BF16的指数部分全部设为同一个值（也即构造一个极端情况），看看能否正常 
        set_BF16_e((uint16_t*)srcBuf, srcSize/2);
        
        /* 3. 调用压缩函数 */
        DataType dt = DT_BF16;
        size_t compBytes = HUF_compress_float_fixRatio(
            compBuf,        /* 压缩后数据存储位置 */
            compBufSize,    /* 压缩buffer最大大小 */
            srcBuf,         /* 原始数据 */
            srcSize,        /* 原始数据大小 */
            R145,           /* 压缩比率 */             // 可以在这里修改 R139, R145, R152 等值
            dt              /* 数据类型 */
        );
        
        if (compBytes == 0 || compBytes > 0xFFFFFFFFUL) {
            printf("Error: Compression failed\n");
            free(compBuf);
            free(srcBuf);
            return -1;
        }
        
        printf("Compressed size: %zu bytes (ratio: %.2f%%)\n", compBytes, (compBytes * 100.0) / srcSize);
        
        /* 4. 准备解压buffer */
        uint8_t* decompBuf = (uint8_t*)malloc(srcSize);
        if (!decompBuf) {
            printf("Error: Failed to allocate memory for decompression buffer\n");
            free(compBuf);
            free(srcBuf);
            return -1;
        }
        
        /* 5. 调用解压函数 */
        DataType decomp_dt;
        size_t decompBytes = HUF_decompress_float_fixRatio(
            decompBuf,      /* 解压后数据存储位置 */
            srcSize,        /* 解压buffer大小（应等于原始数据大小） */
            compBuf,        /* 压缩数据 */
            compBytes,      /* 压缩数据大小 */
            &decomp_dt      /* 数据类型 */
        );

        assert(dt == decomp_dt);  // 检查压缩前后数据类型相同
        
        if (decompBytes != srcSize || decompBytes > 0xFFFFFFFFUL) {
            printf("Error: Decompression failed or size mismatch (expected %zu, got %zu)\n", srcSize, decompBytes);
            free(decompBuf);
            free(compBuf);
            free(srcBuf);
            return -1;
        }
        
        /* 6. 比较原始数据和解压数据 */
        compare_BF16((uint16_t*)srcBuf, (uint16_t*)decompBuf, srcSize/2);
            
        /* 7. 清理资源 */
        free(decompBuf);
        free(compBuf);
        free(srcBuf);
        
        fclose(fp);
    }
    return 0;
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_bf16_tensor.bin>\n", argv[0]);
        return 1;
    }
    
    return test_huf_compression(argv[1]);
}
