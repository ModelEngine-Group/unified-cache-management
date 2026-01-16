#ifndef __TUNSTALL_H__
#define __TUNSTALL_H__

#include <stddef.h>
#include <stdint.h>

//---------------------------------------------------------------------------------------------------------------------------------------------------------
// 函数返回(错误
//---------------------------------------------------------------------------------------------------------------------------------------------------------
#define R_TS_OK 0
#define R_ERR_UNSUPPORT 1          // 超参数不支持
#define R_ERR_SYNTAX 2             // 待解压的流中有格式错误
#define R_ERR_SYMB_RANGE 3         // 输入数组符号范围超限
#define R_ERR_SYMB_RANGE_PREDEF 4  // 输入数组符号范围超限 (预设表模式下)
#define R_ERR_DST_OVERFLOW 5
#define R_ERR_SRC_OVERFLOW 6
#define R_ERR_LUT_BUILD 7
#define R_ERR_LUT_CHECK 8
#define R_ERR_LUT_MISMATCH 9
#define R_ERR_NORMALIZE 10
#define R_ERR_PREDEF_UNINIT 11  // 预设表尚未初始化
#define R_ERR_LARGER 12

//---------------------------------------------------------------------------------------------------------------------------------------------------------
// tunstall 算法超参数
//---------------------------------------------------------------------------------------------------------------------------------------------------------
#define TS_SYMB_BITS 6                 // 每个符号有多少个 Bit
#define TS_N_SYMB (1 << TS_SYMB_BITS)  // 支持的符号取值范围为 0 ~ (TS_N_SYMB-1)
#define TS_N_SYMB_PREDEF 32            // 预设表模式支持的最大符号数据
#define TS_COUNT_PREDEF 64             // 有几个预设表
#define TS_MARK_BITS 8                 // 压缩流中，每个tunstall mark 占几个bit 。目前只支持 8 和 12
#define TS_ITEM_SIZE 7                 // 每个 tunstall LUT item 最多存放几symb ?

#define TS_MODE_UNCOMPRESS 0x00                                      // 非压缩模式
#define TS_MODE_DYNAMIC 0x01                                         // 动态表模式
#define TS_MODE_PREDEF_START 0x02                                    // 预设表模式
#define TS_MODE_PREDEF_END (TS_MODE_PREDEF_START + TS_COUNT_PREDEF)  //

//---------------------------------------------------------------------------------------------------------------------------------------------------------
// struct定义
//---------------------------------------------------------------------------------------------------------------------------------------------------------

// 直方图表项，以数组的形式放在 ts_hist_t
typedef struct {
    uint8_t symb;     // 符号值(符号=未压缩数据流中的符号)
    uint64_t freq;    // 符号频率
    double prob;      // 符号概率，也即归一化的频率
    double log_prob;  // log_probs = log2(1.0 / prob) , 提前计算好放在这里
                      // 用来快速计算预设表和待压缩数据的交叉熵
} ts_hist_item_t;

// 对输入数据的统计直方图
typedef struct {
    ts_hist_item_t
        hist[TS_N_SYMB];  // 直方图，其中的每项是 {symb, freq, prob} 按照 freq 从大到小排序
    int n_symb;           // 非零symbol 总数
} ts_hist_t;

// LUT 表项。实际使用时，是 TS_LUT_SIZE 个表项构成一个表项数据
typedef struct {
    uint8_t v[TS_ITEM_SIZE];  // v[0], v[1], ... 是表项对应的符号值
    uint8_t c;                // 该表项中有多少个有效符号值
} ts_lut_item_t;

// state trans table for tunstall encoding
typedef struct {
    uint16_t next_state[TS_N_SYMB];
    uint16_t mark;
} ts_enc_state_item_t;

#define TS_LUT_SIZE (1 << TS_MARK_BITS)  // LUT 表项数量，也即tunstall mark 的取值范围
#define TS_LUT_BYTES (TS_LUT_SIZE * sizeof(ts_lut_item_t))  // LUT 表的总大小(Bytes)
#define TS_ENC_STATE_TABLE_SIZE (TS_LUT_SIZE * 2)  // enc table 最大有多少个ts_enc_state_item_t
#define TS_ENC_STATE_TABLE_BYTES \
    (TS_ENC_STATE_TABLE_SIZE * sizeof(ts_enc_state_item_t))  // enc table 最大的大小 (Bytes)

// 预设表数据结构，包含 TS_COUNT_PREDEF 预设表
typedef struct {
    ts_hist_t hist[TS_COUNT_PREDEF];
    ts_lut_item_t lut[TS_COUNT_PREDEF][TS_LUT_SIZE];  // 使用这个概率分布建立LUT, 用来解压
    ts_enc_state_item_t etree[TS_COUNT_PREDEF][TS_ENC_STATE_TABLE_SIZE];  // 用来压缩
    int initialized;
} TunstallPredefTables_t;

// 压缩流头
#pragma pack(push, 1)
typedef union {
    // mode = TS_MODE_UNCOMPRESS
    struct {
        uint8_t mode;
        uint32_t length;
    } uncompress;

    // mode = TS_MODE_DYNAMIC
    struct {
        uint8_t mode;
        uint32_t n_mark;
        uint32_t src_len;
        ts_lut_item_t lut[TS_LUT_SIZE];
    } dynamic;

    // mode = TS_MODE_PREDEF_START ~ TS_MODE_PREDEF_END-1
    struct {
        uint8_t mode;
        uint8_t n_symb;
        uint32_t n_mark;   // single-stream mark count
        uint32_t src_len;  // single-stream decompressed length
        uint8_t idx2symb[TS_N_SYMB_PREDEF];
    };

} ts_header_t;
#pragma pack(pop)

//---------------------------------------------------------------------------------------------------------------------------------------------------------
// 函数定义
//---------------------------------------------------------------------------------------------------------------------------------------------------------
extern int TunstallInitAllPredefTables(TunstallPredefTables_t* p_predef_luts);
extern int TunstallCompressDynamic(uint8_t* p_dst, size_t* p_dst_len, const uint8_t* p_src,
                                   size_t src_len);
extern int TunstallCompressPredef(uint8_t* p_dst, size_t* p_dst_len, const uint8_t* p_src,
                                  size_t src_len, const TunstallPredefTables_t* p_predef_luts);
extern int TunstallCompress(uint8_t* p_dst, size_t* p_dst_len, const uint8_t* p_src, size_t src_len,
                            const TunstallPredefTables_t* p_predef_luts);
extern int TunstallDecompress(uint8_t* p_dst, size_t* p_dst_len, const uint8_t* p_src,
                              int do_idx2symb_on_predef_lut,
                              const TunstallPredefTables_t* p_predef_luts);

#endif  //__TUNSTALL_H__
