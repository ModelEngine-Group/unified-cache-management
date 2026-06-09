
#include <limits.h>  // from limits.h import UINT32_MAX
#include <math.h>    // from math.h import log
#include <stdlib.h>  // from stdlib.h import qsort
#include <string.h>  // from string.h import memset, memcpy
#if TS_DEBUG_PRINT
#include <stdio.h>
#endif

#include "tunstall.h"

#define TS_V2_EXPAND \
    1  // 使用 V2 版算法来进行 tunstall LUT 展开 。当剩余 prob
       // 小于阈值时停止展开低概率符号，避免浪费LUT条目
#ifndef TS_V2_THRESH
#define TS_V2_THRESH 1.2  // V2 算法参数（可通过编译选项 -DTS_V2_THRESH=... 覆盖）
#endif
#define TS_DEBUG_PRINT 0

#define RET_ERROR_IF(err_code, condition) \
    {                                     \
        int c = (condition);              \
        int e = (err_code);               \
        if (c) { return e; }              \
    }
#define RET_WHEN_ERROR(err_code) \
    {                            \
        int e = (err_code);      \
        if (e) { return e; }     \
    }

// 获取 header 占用的字节数
static size_t ts_get_header_length(const ts_header_t* p_hdr)
{
    uint8_t mode = p_hdr->mode;
    if (mode == TS_MODE_UNCOMPRESS) {
        return sizeof(p_hdr->uncompress);
    } else if (mode == TS_MODE_DYNAMIC) {
        return sizeof(p_hdr->dynamic);
    } else if (TS_MODE_PREDEF_START <= mode && mode < TS_MODE_PREDEF_END) {
        return sizeof(p_hdr->mode) + sizeof(p_hdr->n_symb) + sizeof(p_hdr->n_mark) +
               sizeof(p_hdr->src_len) + p_hdr->n_symb;
    } else {
        return (size_t)(-1);
    }
}

// 功能：不压缩直接存数据
static int ts_do_not_compress(uint8_t* p_dst, size_t* p_dst_len, const uint8_t* p_src,
                              size_t src_len)
{
    uint8_t* p_dst_base = p_dst;  // 整个压缩流的起始地址
    ts_header_t* p_hdr =
        (ts_header_t*)p_dst;  // 压缩流头直接指向压缩流的开始，后续构建 p_hdr ，避免搬移
    RET_ERROR_IF(R_ERR_DST_OVERFLOW,
                 (src_len + sizeof(p_hdr->uncompress) > *p_dst_len));  // 如果dst塞不下，报错
    p_hdr->mode = TS_MODE_UNCOMPRESS;                                  // 调整非压缩模式
    p_hdr->uncompress.length = src_len;
    p_dst = p_dst_base + ts_get_header_length(p_hdr);
    memcpy(p_dst, p_src, src_len);
    *p_dst_len = (p_dst + src_len) - p_dst_base;  // dst大小
    return R_TS_OK;
}

// 建立 idx -> symb 的映射表 (idx2symb)
// 建立 symb -> idx 的映射表 (symb2idx)
static void ts_build_idx2symb_symb2idx(uint8_t* p_idx2symb, uint8_t* p_symb2idx,
                                       const ts_hist_t* p_hist)
{
    for (int idx = 0; idx < p_hist->n_symb;
         idx++) {  // 这里 symb 循环变量类型要取 int 而不uint8_t ，避免死循环
        uint8_t symb = p_hist->hist[idx].symb;
        p_idx2symb[idx] = symb;
        p_symb2idx[symb] = idx;
    }
}

// 功能: 使用函数构建预设表的归一化概率分布表 (PDF) ，也即ts_predef_lut_t->probs
static int ts_init_predef_probs(ts_hist_t* p_hist, int n_symb, int lambda)
{
    p_hist->n_symb = TS_N_SYMB_PREDEF;
    for (int idx = 0; idx < n_symb; idx++) {
        p_hist->hist[idx].symb = idx;
        p_hist->hist[idx].prob =
            exp(-((lambda + 1.0) / 64.0) * idx);  // 函数表达式f[x] = exp ( - ((λ+1)/64) * x)
    }
    double sum = 0.0;
    for (int idx = 0; idx < n_symb; idx++) {  // 求和
        sum += p_hist->hist[idx].prob;
    }
    RET_ERROR_IF(R_ERR_NORMALIZE, sum <= 0.0);
    double neg_inv_loge2 = -1.0 / log(2.0);
    for (int idx = 0; idx < n_symb; idx++) {  // 归一化
        p_hist->hist[idx].prob /= sum;
        p_hist->hist[idx].log_prob = log(p_hist->hist[idx].prob) *
                                     neg_inv_loge2;  // log_prob = log_e(1.0 / prob) / log_e(2.0);
    }
    return R_TS_OK;
}

// 功能：计算待压缩的数据的实际分布 (p_hist) 在某个预设LUT (p_pdlut)
// 下的交叉熵，交叉熵越小，说明越适合用这个预设表来压缩
static double ts_cross_entropy(const ts_hist_t* p_hist1, const ts_hist_t* p_hist2)
{
    if (p_hist1->n_symb > p_hist2->n_symb) {  // 实际数据的取值范围> 预设LUT的取值范围
        return -1.0;                          // 返回非法值，无法使用这个预设LUT来压缩
    } else {
        double entropy = 0.0;
        for (int idx = 0; idx < p_hist1->n_symb; idx++) {
            entropy += p_hist1->hist[idx].prob *
                       p_hist2->hist[idx].log_prob;  // 交叉熵= p1 * log2(1.0 / p2)
        }
        return entropy;
    }
}

// 功能：比较两ts_hist_item_t ，用于对直方图进qsort 排序
static int ts_cmp_hist(const void* p_a, const void* p_b)
{
    uint64_t a_freq = ((ts_hist_item_t*)p_a)->freq;
    uint64_t b_freq = ((ts_hist_item_t*)p_b)->freq;
    uint8_t a_symb = ((ts_hist_item_t*)p_a)->symb;
    uint8_t b_symb = ((ts_hist_item_t*)p_b)->symb;
    if (a_freq != b_freq) {                            // 如果频率不同
        return (a_freq < b_freq) - (a_freq > b_freq);  // 频率高的排在前面
    } else {                                           // 如果频率相同
        return (a_symb > b_symb) - (a_symb < b_symb);  // 符号值小的排在前面
    }
}

// 功能：统计输入数据中各符号的出现频率，构建直方图（也即概率表）同时检查符号值是否在有效范围
static int ts_get_hist_from_array(ts_hist_t* p_hist, const uint8_t* p_src, size_t src_len)
{
    memset(p_hist, 0, sizeof(ts_hist_t));
    for (size_t i = 0; i < src_len; i++) {  // 循环：遍历输入数组，统计频率freq
        uint8_t symb = p_src[i];
        RET_ERROR_IF(R_ERR_SYMB_RANGE, symb >= TS_N_SYMB);  // 检查符号是否超出范围
        p_hist->hist[symb].freq++;                          // 对应符号频率
    }
    for (int symb = 0; symb < TS_N_SYMB;
         symb++) {  // 循环：遍历所有符号，把频率freq转化为概率prob 这里 symb 循环变量类型要取 int
                    // 而不uint8_t ，避免死循环
        uint64_t freq = p_hist->hist[symb].freq;
        p_hist->hist[symb].symb = symb;
        p_hist->hist[symb].prob = (double)freq / src_len;  // 概率 = 频率 / 总数
        if (freq) { p_hist->n_symb++; }                    // 遇到非零 symbolsymbol 总数 +1
    }
    qsort((void*)(p_hist->hist), TS_N_SYMB, sizeof(ts_hist_item_t),
          ts_cmp_hist);  // hist 排序, 频率高的排在前面
    return R_TS_OK;
}

// 功能：打印各符号的统计频率和概率值，用于调试分析
#if TS_DEBUG_PRINT
static void ts_print_hist(const ts_hist_t* p_hist)
{
    printf("------ start of probs ----------------------------------------------\n");
    printf("n_symb = %3d\n", p_hist->n_symb);
    for (int idx = 0; idx < p_hist->n_symb; idx++) {
        uint8_t symb = p_hist->hist[idx].symb;
        uint64_t freq = p_hist->hist[idx].freq;
        double prob = p_hist->hist[idx].prob;
        printf("symb=%2d   freq=%9d   prob=%.6lf \n", symb, freq, prob);
    }
    printf("\n------ end of probs ----------------------------------------------\n\n");
}
#endif

// 功能：创建新的LUT表项，将新符号添加到已有表项的末尾，返回新的表项副本
static ts_lut_item_t ts_add_symb_to_item(const ts_lut_item_t* p_item, uint8_t new_symb)
{
    ts_lut_item_t item = *p_item;
    item.v[item.c] = new_symb;  // 在新位置添加符号
    item.c += 1;                // 符号计数据
    return item;
}

// 功能：在LUT中搜索概率最大的表项，排除黑名单中的项，也要排除达到符号数量上限(TS_ITEM_SIZE)的项。返回项的编号（也即mark
static uint32_t ts_search_largest_item(const ts_lut_item_t* p_lut, const double* p_lut_prob,
                                       const uint8_t* p_black_list, uint32_t lut_size)
{
    uint32_t max_mark = UINT32_MAX;
    double max_prob = 0.0;
    for (uint32_t mark = 0; mark < lut_size; mark++) {
        if (p_black_list == NULL || !p_black_list[mark]) {  // 黑名单不存在，或者该条目不在黑名单中
            uint8_t cnt = p_lut[mark].c;                    // LUT本项中的符号数量
            if (cnt < TS_ITEM_SIZE) {                       // 且该条目的符号数量没到上
                if (max_prob < p_lut_prob[mark]) {          // 找到更大小prob
                    max_prob = p_lut_prob[mark];
                    max_mark = mark;
                }
            }
        }
    }
    return max_mark;  // 返回 prob 最大的可展开项索
}

// 功能: 检查LUT
static int ts_check_lut(const ts_lut_item_t* p_lut)
{
    uint32_t mark = 0;
    for (; mark < TS_LUT_SIZE; mark++) {
        uint8_t cnt = p_lut[mark].c;
        RET_ERROR_IF(R_ERR_LUT_CHECK, cnt > TS_ITEM_SIZE);  // 无效项检
        if (cnt == 0) { break; }
    }
    for (; mark < TS_LUT_SIZE; mark++) {
        uint8_t cnt = p_lut[mark].c;
        RET_ERROR_IF(R_ERR_LUT_CHECK, cnt != 0);
    }
    return R_TS_OK;
}

// 功能：沿着一ts_lut_item_t 中包含的字符，从 p_enc_table 的根节点摸到一个节
static uint16_t ts_goto_state_by_lut_item(const ts_enc_state_item_t* p_enc_table,
                                          const ts_lut_item_t* p_item)
{
    uint16_t state = 0;
    for (int i_symb = 0; i_symb < p_item->c; i_symb++) {
        uint8_t symb = p_item->v[i_symb];
        state = p_enc_table[state].next_state[symb];
    }
    return state;
}

// 功能: tunstall LUT 展开算法
//       每次LUT 中删除概率最大的 item, 并遍历所有非symb , 给这里item 末尾拼上这个 symb
//       作为最新的一item 加入 LUT 顺便也会建立状态转移树p_enc_table , 用来压缩
static int ts_build_lut(ts_lut_item_t* p_lut,
                        ts_enc_state_item_t* p_enc_table,  // 建立状态转移树，用来进tunstall 压缩
                        const ts_hist_t* p_hist)
{
    int n_symb = p_hist->n_symb;

    double lut_prob[TS_LUT_SIZE] = {1.0};  // 数组：记录各 item 的条件概率。其init_mark 的概= 1.0
    uint8_t lut_black_list[TS_LUT_SIZE] = {
        0};  // 数组：黑名单，上了黑名单的LUT表项不能被展开，初始化全都不在黑名

    memset(p_lut, 0, TS_LUT_BYTES);  // 清空 LUT
    uint32_t lut_size = 1;           // 初始只有一item，其 mark = 0 (init_mark)

    memset(p_enc_table, 0, TS_ENC_STATE_TABLE_BYTES);
    uint16_t new_state = 0;   // 当前最新的state, 最开始只有一个状
    p_enc_table[0].mark = 0;  // init_mark = 0

    do {
        uint32_t mark = ts_search_largest_item(p_lut, lut_prob, lut_black_list,
                                               lut_size);  // 找概率最大的可展开项。并排除黑名

        if (mark == UINT32_MAX) {  // printf("Warning: cannot expand more, LUT size is only %u\n",
                                   // lut_size);
            break;                 // 所有项目都在黑名单中了，无法再展开
        }

        double max_prob = lut_prob[mark];      // 概率最大的可展开项的概率
        ts_lut_item_t max_item = p_lut[mark];  // 概率最大的可展开项的 item, 也即待展开item
        p_lut[mark].c = 0;                     // 原位置标记为无效 LUT item

        uint16_t state = ts_goto_state_by_lut_item(
            p_enc_table,
            &max_item);  // 找到 max_item (也即需要展开的item) 对应的state：沿着一ts_lut_item_t
                         // 中包含的字符，从 p_enc_table 的根节点摸到一个节

        double rem_prob = 1.0;  // 记录剩余概率, 用于阈值判

        for (int idx = 0; idx < n_symb; idx++) {        // 遍历所symb, 逐个展开
            uint8_t symb = p_hist->hist[idx].symb;      // symb
            double symb_prob = p_hist->hist[idx].prob;  // symb 概率

            while (p_lut[mark].c != 0) {  // 跳到下一个空LUT 表项
                mark++;
                RET_ERROR_IF(R_ERR_LUT_BUILD, mark >= TS_LUT_SIZE);
            }

            if (mark < TS_LUT_SIZE - 1) {  // 如果 LUT 还没被填
#if TS_V2_EXPAND                       // 是否使用 V2 版算法来进行 tunstall LUT 展开 。当剩余 prob
                                       // 小于阈值时停止展开低概率符号，避免浪费LUT条目
                if (max_item.c > 0) {  // V2关键特 "中断展开" 注意: 只有非根节点允许"中断展开",
                                       // 所以这里要判断 (max_item.c > 0)
                    if (0 < idx && idx < (n_symb - 1)) {  // 待展开symb 不是第一个，也不是最后一
                        if (max_prob * rem_prob <
                            (TS_V2_THRESH /
                             TS_LUT_SIZE)) {           // 剩余未展开symb 的总概率是否小于一个阈值？
                            p_lut[mark] = max_item;    //   剩余符号不再展开，把旧符号写入当前项
                            lut_black_list[mark] = 1;  //   把当前项加入黑名单，禁止后续展开
                            p_enc_table[state].mark = mark;
                            break;  //   中断当前项的展开
                        }
                    }
                }
#endif

                p_lut[mark] = ts_add_symb_to_item(&max_item, symb);  // 添加新符号创建新
                lut_prob[mark] = max_prob * symb_prob;               // 计算prob（条件概率）
                new_state++;                                         // 状态转移表+1
                p_enc_table[state].next_state[symb] =
                    new_state;  // 建立状态转state(symb) -> new_state
                p_enc_table[new_state].mark = mark;

            } else if (idx >= (n_symb - 1)) {  // 如果 LUT 被填(当前 mark 已经对应最后一个表项了) ,
                                               // 而且当前面item 刚好被展开完了
                p_lut[mark] = ts_add_symb_to_item(&max_item, symb);  // 添加新符号创建新
                new_state++;                                         // 状态转移表+1
                p_enc_table[state].next_state[symb] =
                    new_state;  // 建立状态转state(symb) -> new_state
                p_enc_table[new_state].mark = mark;
                break;
            } else {  // 如果 LUT 被填(当前 mark 已经对应最后一个表项了) , 但是当前面item
                      // 没被完全展开
                p_lut[mark] = max_item;  // 旧表项写入最后一个表
                p_enc_table[state].mark = mark;
                break;
            }

            rem_prob -= symb_prob;  // rem_prob 是剩余未展开的符号的总概
        }

        lut_size = mark + 1;  // 更新当前LUT大小
    } while (lut_size < TS_LUT_SIZE);

    return R_TS_OK;
}

// 功能：判断预设表是否可用
static int ts_all_predef_luts_available(const TunstallPredefTables_t* p_predef_luts)
{
    return (p_predef_luts != NULL) && (p_predef_luts->initialized == 1);
}

// 功能：初始化预设表。程序运行前调用一---------------------------------------
int TunstallInitAllPredefTables(TunstallPredefTables_t* p_predef_luts)
{
    for (int lambda = 0; lambda < TS_COUNT_PREDEF; lambda++) {
        RET_WHEN_ERROR(
            ts_init_predef_probs(&(p_predef_luts->hist[lambda]), TS_N_SYMB_PREDEF, lambda));
        RET_WHEN_ERROR(ts_build_lut(p_predef_luts->lut[lambda], p_predef_luts->etree[lambda],
                                    &(p_predef_luts->hist[lambda])));
        RET_WHEN_ERROR(
            ts_check_lut(p_predef_luts->lut[lambda]));  // 检查刚建立LUT 表是否符合一些基本要
    }
    p_predef_luts->initialized = 1;
    return R_TS_OK;
}

// 功能：针对待压缩的数据分p_hist, 在所有预设表中，寻找最合适（交叉熵最小）的预设表
static int ts_find_best(const TunstallPredefTables_t* p_predef_luts, const ts_hist_t* p_hist,
                        uint8_t* p_best_lambda)
{
    RET_ERROR_IF(R_ERR_PREDEF_UNINIT, !ts_all_predef_luts_available(p_predef_luts));
    double best_entropy = 99999.0;
    *p_best_lambda = 0;
    for (int lambda = 0; lambda < TS_COUNT_PREDEF; lambda++) {
        double entropy = ts_cross_entropy(p_hist, &(p_predef_luts->hist[lambda]));
        if (entropy <= 0.0) { return R_ERR_NORMALIZE; }
        if (best_entropy > entropy) {
            best_entropy = entropy;
            *p_best_lambda = lambda;
        }
    }
    return R_TS_OK;
}

// 功能：打印LUT表中各条目的 prob 和对应的符号序列，用于调
#if TS_DEBUG_PRINT
static void ts_print_lut(const ts_lut_item_t* p_lut)
{
    printf("------ start of LUT ----------------------------------------------\n");
    for (uint32_t i = 0; i < TS_LUT_SIZE; i++) {
        uint8_t cnt = p_lut[i].c;
        printf("[%d]  cnt=%d", i, cnt);
        for (int j = 0; j < cnt; j++) { printf("  %d", p_lut[i].v[j]); }
        printf("\n");
    }
    printf("------ end of LUT ----------------------------------------------\n\n");
}
#endif

// 功能：根idx2symb 映射 LUT 中的 idx 转化symb (异地操作)
static void ts_cvt_lut_idx2symb(ts_lut_item_t* p_lut_dst, const ts_lut_item_t* p_lut_src,
                                const uint8_t* p_idx2symb)
{
    for (uint32_t mark = 0; mark < TS_LUT_SIZE; mark++) {
        p_lut_dst[mark].c = p_lut_src[mark].c;
        for (int i = 0; i < TS_ITEM_SIZE; i++) {
            p_lut_dst[mark].v[i] = p_idx2symb[p_lut_src[mark].v[i]];
        }
    }
}

// 功能：根idx2symb 映射 把一个数组中idx 转化symb (原地操作)
static void ts_cvt_data_idx2symb(uint8_t* p_data, size_t src_len, const uint8_t* p_idx2symb)
{
    for (size_t i = 0; i < src_len; i++) { p_data[i] = p_idx2symb[p_data[i]]; }
}

// 功能：沿着 p_src 的字符，p_enc_table 的根节点摸到一个叶节点，返回叶节点mark
static uint16_t ts_get_mark_by_input_stream(const ts_enc_state_item_t* p_enc_table,
                                            const uint8_t** ipp, const uint8_t* ip_end,
                                            uint8_t most_freq_symb)
{
    uint16_t state = 0;
    for (;;) {
        uint8_t symb = most_freq_symb;  // most_freq_symb 是最频繁symb
        if (*ipp < ip_end) { symb = **ipp; }
        uint16_t mark = p_enc_table[state].mark;
        state = p_enc_table[state].next_state[symb];
        if (state == 0) { return mark; }
        (*ipp)++;
    }
}

// 功能：使用构建好LUT 表对输入数据进行 Tunstall 编码。每次查找最长匹配，将两个标记打包输
static int ts_encode_dynamic(const ts_enc_state_item_t* p_enc_table, uint8_t** opp, uint8_t* op_end,
                             const uint8_t* ip, size_t src_len, uint8_t most_freq_symb,
                             uint32_t* p_n_mark)
{
    const uint8_t* ip_end = ip + src_len;
    *p_n_mark = 0;

    while (ip < ip_end) {
        uint32_t mark1 = ts_get_mark_by_input_stream(p_enc_table, &ip, ip_end, most_freq_symb);
        uint32_t mark2 = ts_get_mark_by_input_stream(p_enc_table, &ip, ip_end, most_freq_symb);
        RET_ERROR_IF(R_ERR_DST_OVERFLOW, (((*opp) + (TS_MARK_BITS * 2 / 8)) >= op_end));
        *(uint32_t*)(*opp) = (mark2 << TS_MARK_BITS) | mark1;
        (*opp) += (TS_MARK_BITS * 2 / 8);
        (*p_n_mark) += 2;
    }

    return R_TS_OK;
}

// 功能：沿着 p_src 的字符，p_enc_table 的根节点摸到一个叶节点，返回叶节点mark
static uint16_t ts_get_mark_by_input_stream_symb2idx(const ts_enc_state_item_t* p_enc_table,
                                                     const uint8_t* p_symb2idx, const uint8_t** ipp,
                                                     const uint8_t* ip_end)
{
    uint16_t state = 0;
    for (;;) {
        uint8_t symb = 0;
        if (*ipp < ip_end) { symb = p_symb2idx[**ipp]; }
        uint16_t mark = p_enc_table[state].mark;
        state = p_enc_table[state].next_state[symb];
        if (state == 0) { return mark; }
        (*ipp)++;
    }
}

// 功能：使用构建好LUT 表对输入数据进行 Tunstall 编码。每次查找最长匹配，将两个标记打包输
static int ts_encode_predef(const ts_enc_state_item_t* p_enc_table, const uint8_t* p_symb2idx,
                            uint8_t** opp, uint8_t* op_end, const uint8_t* ip, size_t src_len,
                            uint32_t* p_n_mark)
{
    const uint8_t* ip_end = ip + src_len;
    *p_n_mark = 0;

    while (ip < ip_end) {
        uint32_t mark1 = ts_get_mark_by_input_stream_symb2idx(p_enc_table, p_symb2idx, &ip, ip_end);
        uint32_t mark2 = ts_get_mark_by_input_stream_symb2idx(p_enc_table, p_symb2idx, &ip, ip_end);
        RET_ERROR_IF(R_ERR_DST_OVERFLOW, (((*opp) + (TS_MARK_BITS * 2 / 8)) >= op_end));
        *(uint32_t*)(*opp) = (mark2 << TS_MARK_BITS) | mark1;
        (*opp) += (TS_MARK_BITS * 2 / 8);
        (*p_n_mark) += 2;
    }

    return R_TS_OK;
}

//---------------------------------------------------------------------------------------------------------------------------------------------------------
// 从这里开始是外部可调用的函数
//---------------------------------------------------------------------------------------------------------------------------------------------------------

// 功能：Tunstall压缩主函数。构建直方图、生成LUT表、执行编码，返回压缩后的数据
int TunstallCompressDynamic(uint8_t* p_dst, size_t* p_dst_len, const uint8_t* p_src, size_t src_len)
{
    RET_ERROR_IF(R_ERR_UNSUPPORT, TS_MARK_BITS != 8 && TS_MARK_BITS != 12);  // 参数检
    RET_ERROR_IF(R_ERR_DST_OVERFLOW, *p_dst_len < TS_LUT_BYTES);  // 目标缓冲区必须能容纳LUT
    RET_ERROR_IF(R_ERR_SRC_OVERFLOW, src_len == 0 || src_len > 0xFFFFFFFF);  // 源缓冲区长度检
    uint8_t* p_dst_base = p_dst;                                             // 整个压缩流的起始地址
    ts_header_t* p_hdr =
        (ts_header_t*)p_dst;  // 压缩流头直接指向压缩流的开始，后续构建 p_hdr ，避免搬移

    ts_hist_t hist;                                                 // 统计信息
    RET_WHEN_ERROR(ts_get_hist_from_array(&hist, p_src, src_len));  // 统计符号频率, 得到直方图
    uint8_t most_freq_symb = hist.hist[0].symb;

    ts_enc_state_item_t
        etree[TS_ENC_STATE_TABLE_SIZE];  // TODO : TS_N_SYMB 太大小 etree 占空间太大，栈放不下
    RET_WHEN_ERROR(
        ts_build_lut(p_hdr->dynamic.lut, etree, &hist));  // 建立 LUT, 直接建立p_hdr->dynamic.lut
    RET_WHEN_ERROR(ts_check_lut(p_hdr->dynamic.lut));     // 检LUT 表是否符合一些基本要
    p_hdr->mode = TS_MODE_DYNAMIC;
    p_hdr->dynamic.src_len =
        static_cast<uint32_t>(src_len);  // 标记为动态表模式，让解压器能识别模式

#if TS_DEBUG_PRINT                     //
    ts_print_lut(p_hdr->dynamic.lut);  // 打印编码
    ts_print_hist(&hist);              // 打印符号统计
#endif                                 //

    p_dst += ts_get_header_length(p_hdr);  // p_dst 跳过 header (在动态表模式 header 就是动LUT
    RET_WHEN_ERROR(ts_encode_dynamic(etree, &p_dst, (p_dst_base + *p_dst_len), p_src, src_len,
                                     most_freq_symb,
                                     &p_hdr->dynamic.n_mark));  // 执行 tunstall 编码

    *p_dst_len = p_dst - p_dst_base;  // 压缩流长 总长

    return R_TS_OK;
}

// 功能：Tunstall压缩主函数。构建直方图、生成LUT表、执行编码，返回压缩后的数据
int TunstallCompressPredef(uint8_t* p_dst, size_t* p_dst_len, const uint8_t* p_src, size_t src_len,
                           const TunstallPredefTables_t* p_predef_luts)
{
    RET_ERROR_IF(R_ERR_PREDEF_UNINIT, !ts_all_predef_luts_available(p_predef_luts));
    RET_ERROR_IF(R_ERR_UNSUPPORT, TS_MARK_BITS != 8 && TS_MARK_BITS != 12);  // 参数检
    RET_ERROR_IF(R_ERR_DST_OVERFLOW, *p_dst_len < TS_LUT_BYTES);  // 目标缓冲区必须能容纳LUT
    RET_ERROR_IF(R_ERR_SRC_OVERFLOW, src_len == 0 || src_len > 0xFFFFFFFF);  // 源缓冲区长度检
    uint8_t* p_dst_base = p_dst;                                             // 整个压缩流的起始地址
    ts_header_t* p_hdr =
        (ts_header_t*)p_dst;  // 压缩流头直接指向压缩流的开始，后续构建 p_hdr ，避免搬移

    ts_hist_t hist;                                                 // 统计信息
    RET_WHEN_ERROR(ts_get_hist_from_array(&hist, p_src, src_len));  // 统计符号频率, 得到直方图
    RET_ERROR_IF(
        R_ERR_SYMB_RANGE_PREDEF,
        hist.n_symb > TS_N_SYMB_PREDEF);  // 如果非零符号数量 > 预设表模式所允许的最大非零符号数据

    uint8_t lambda;
    RET_WHEN_ERROR(ts_find_best(p_predef_luts, &hist, &lambda));  // 按交叉熵选择最优预设表

    p_hdr->mode = TS_MODE_PREDEF_START + lambda;
    p_hdr->n_symb = hist.n_symb;
    p_hdr->src_len = static_cast<uint32_t>(src_len);
    uint8_t symb2idx[TS_N_SYMB];
    ts_build_idx2symb_symb2idx(p_hdr->idx2symb, symb2idx, &hist);  // 建立 idx2symb/symb2idx 映射

#if TS_DEBUG_PRINT         //
    ts_print_hist(&hist);  // 打印符号统计
#endif                     //

    p_dst += ts_get_header_length(p_hdr);  // p_dst 跳过 header
    RET_WHEN_ERROR(ts_encode_predef(p_predef_luts->etree[lambda], symb2idx, &p_dst,
                                    (p_dst_base + *p_dst_len), p_src, src_len,
                                    &p_hdr->n_mark));  // 执行 tunstall 编码

    *p_dst_len = p_dst - p_dst_base;  // 压缩流长 总长

    return R_TS_OK;
}

// 功能：Tunstall压缩主函数。在 dynamic, predef, 非压缩模式中三选一
int TunstallCompress(uint8_t* p_dst, size_t* p_dst_len, const uint8_t* p_src, size_t src_len,
                     const TunstallPredefTables_t* p_predef_luts)
{
    size_t len_dynam = *p_dst_len;
    size_t len_predef = *p_dst_len;
    int err_dynam = TunstallCompressDynamic(p_dst, &len_dynam, p_src, src_len);  // 尝试dynmaic 压缩
    int err_predef = TunstallCompressPredef(p_dst, &len_predef, p_src, src_len,
                                            p_predef_luts);  // 尝试predef 压缩
    if (!err_dynam && len_dynam >= src_len) {  // 如果 dynamic 虽然没出错，但是压缩后比原始更长
        err_dynam = R_ERR_LARGER;              // 认为 dynamic 出错
    }
    if (!err_predef && len_predef >= src_len) {  // 如果 predef 虽然没出错，但是压缩后比原始更长
        err_predef = R_ERR_LARGER;               // 认为 dynamic 出错
    }
    if (!err_dynam && !err_predef) {    // 如果 dynamic predef 都没出错
        if (len_predef >= len_dynam) {  // 看哪个更长，把更长的模式设置为出
            err_predef = R_ERR_LARGER;
        } else {
            err_dynam = R_ERR_LARGER;
        }
    }
    if (!err_dynam) {  // 最终选择一种模式进行压缩
        return TunstallCompressDynamic(p_dst, p_dst_len, p_src, src_len);  // 采用 dynamic 模式
    } else if (!err_predef) {                                              //
        *p_dst_len = len_predef;  // 采用 predef 模式，这里不需要再压缩了，因为 dst 里面已经放了
                                  // predef 压缩的数据
        return R_TS_OK;           //
    } else {                      //
        return ts_do_not_compress(p_dst, p_dst_len, p_src, src_len);  // 采用非压缩模式
    }
}

// 功能：Tunstall解压缩主函数。根据LUT表将标记还原为原始符号序
int TunstallDecompress(uint8_t* p_dst, size_t* p_dst_len, const uint8_t* p_src,
                       int do_idx2symb_on_predef_lut, const TunstallPredefTables_t* p_predef_luts)
{
    uint8_t* p_dst_base = p_dst;
    const ts_header_t* p_hdr = (ts_header_t*)p_src;

    if (p_hdr->mode == TS_MODE_UNCOMPRESS) {
        p_src += ts_get_header_length(p_hdr);
        memcpy(p_dst, p_src, p_hdr->uncompress.length);
        *p_dst_len = p_hdr->uncompress.length;

    } else {
        ts_lut_item_t lut[TS_LUT_SIZE];
        const ts_lut_item_t* p_lut;
        uint32_t n_mark = 0;
        uint32_t src_len = 0;

        if (p_hdr->mode == TS_MODE_DYNAMIC) {
            p_lut = p_hdr->dynamic.lut;
            RET_WHEN_ERROR(ts_check_lut(p_lut));
            n_mark = p_hdr->dynamic.n_mark;
            src_len = p_hdr->dynamic.src_len;
        } else if (TS_MODE_PREDEF_START <= p_hdr->mode && p_hdr->mode < TS_MODE_PREDEF_END) {
            RET_ERROR_IF(R_ERR_PREDEF_UNINIT, !ts_all_predef_luts_available(p_predef_luts));
            uint8_t lambda = p_hdr->mode - TS_MODE_PREDEF_START;
            if (do_idx2symb_on_predef_lut) {
                ts_cvt_lut_idx2symb(lut, p_predef_luts->lut[lambda], p_hdr->idx2symb);
                p_lut = lut;
            } else {
                p_lut = p_predef_luts->lut[lambda];
            }
            n_mark = p_hdr->n_mark;
            src_len = p_hdr->src_len;
        } else {
            RET_ERROR_IF(R_ERR_SYNTAX, 1);
        }

        p_src += ts_get_header_length(p_hdr);

        for (uint32_t i_mark = 0; i_mark < n_mark; i_mark += 2) {
            uint32_t mark21 = *(uint32_t*)p_src;
            uint32_t mark1 = mark21 & ((1 << TS_MARK_BITS) - 1);
            uint32_t mark2 = (mark21 >> TS_MARK_BITS) & ((1 << TS_MARK_BITS) - 1);
            p_src += (TS_MARK_BITS * 2 / 8);

            memcpy(p_dst, p_lut[mark1].v, sizeof(ts_lut_item_t));
            p_dst += p_lut[mark1].c;

            memcpy(p_dst, p_lut[mark2].v, sizeof(ts_lut_item_t));
            p_dst += p_lut[mark2].c;
        }

        RET_ERROR_IF(R_ERR_SYNTAX, (size_t)(p_dst - p_dst_base) < src_len);
        *p_dst_len = src_len;

        if (p_hdr->mode != TS_MODE_DYNAMIC && !do_idx2symb_on_predef_lut) {
            ts_cvt_data_idx2symb(p_dst_base, (*p_dst_len), p_hdr->idx2symb);
        }
    }

    return R_TS_OK;
}
