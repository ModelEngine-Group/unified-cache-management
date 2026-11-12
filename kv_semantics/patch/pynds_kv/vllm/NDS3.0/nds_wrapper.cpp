// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include <algorithm>
#include <atomic>
#include <cerrno>  // for errno, ERANGE
#include <chrono>
#include <cstdarg>  // for va_list
#include <cstdint>
#include <cstdio>   // for vfprintf
#include <cstdlib>  // for getenv, atoi
#include <cstring>  // for strcmp
#include <limits>   // for std::numeric_limits
#include <memory>
#include <nds_interface.h>
#include <unordered_map>  // for per-nsid valueSize 覆盖表
#include <utility>        // for std::pair（MR 范围表）
#include <vector>

// xxhash（vendored 单头文件，XXH_INLINE_ALL 头文件内联，无需改构建脚本）：
// 用于 80-bit block_key 的默认打散（见 scatter_key）
#define XXH_INLINE_ALL
#include "xxhash.h"

// libndsclient.so 的私有最小 ABI 声明，仅供本实现文件使用。
// 集群模式的连接管理、哈希环、请求分发和断链重连均由库内负责，不在上层
// 重复声明其私有类型或全局状态。以下布局、数组长度和函数签名必须与库一致。
#define MAX_ADDR_NUM_IPC 4096

#pragma pack(push, 1)
typedef struct {
    uint64_t key_low;
    uint16_t key_high;
    uint16_t reserve[3];
} block_key;

typedef struct {
    uint32_t offset;
    block_key block_id;
    uint64_t hbm_addr;
    uint32_t length;
    uint32_t remote_key;
} addr_pair;
#pragma pack(pop)

typedef struct {
    uint64_t task_id;
    uint64_t mr_id;
    uint16_t addr_num;
    uint32_t nsid;
    addr_pair addr_pair_list[MAX_ADDR_NUM_IPC];
} NDS_REQ;

struct kv_mr {
    uint64_t addr;
    uint64_t len;
};

typedef int NDS_ERROR_T;

typedef struct {
    std::vector<int> batch_code_set;
    std::vector<int> key_code_set;
} CODE_SET;

static_assert(sizeof(block_key) == 16, "NDS block_key ABI mismatch");
static_assert(sizeof(addr_pair) == 36, "NDS addr_pair ABI mismatch");
static_assert(sizeof(kv_mr) == 16, "NDS kv_mr ABI mismatch");
static_assert(offsetof(NDS_REQ, task_id) == 0, "NDS_REQ.task_id ABI mismatch");
static_assert(offsetof(NDS_REQ, mr_id) == 8, "NDS_REQ.mr_id ABI mismatch");
static_assert(offsetof(NDS_REQ, addr_num) == 16, "NDS_REQ.addr_num ABI mismatch");
static_assert(offsetof(NDS_REQ, nsid) == 20, "NDS_REQ.nsid ABI mismatch");
static_assert(offsetof(NDS_REQ, addr_pair_list) == 24, "NDS_REQ.addr_pair_list ABI mismatch");
static_assert(sizeof(NDS_REQ) == 24 + MAX_ADDR_NUM_IPC * 36, "NDS_REQ size ABI mismatch");

NDS_ERROR_T nds_init(void** kv_cache, int mr_num);
uint64_t nds_kv_exist(NDS_REQ* req);
NDS_ERROR_T nds_kv_get(NDS_REQ* req);
NDS_ERROR_T nds_kv_put(NDS_REQ* req);
CODE_SET nds_kv_wait(uint64_t task_id, uint32_t total_blocks);
void nds_kv_get_nds_config(const char* path_nds_config);

// 全局配置变量（默认值与原先硬编码一致，可被 nds_wrapper.conf 覆盖）
static size_t g_max_exist_addr_num = 256;
static size_t g_length_per_read_batch = 131072;   // 128K
static size_t g_length_per_write_batch = 131072;  // 128K
static size_t g_length_per_read_req = 33554432;   // 32M
static size_t g_length_per_write_req = 33554432;  // 32M
static size_t g_value_size = 33554432;            // 32M, NDS namespace valueSize（全局默认）

// 每线程滑动窗口大小（conf 键 max_inflight_reqs，默认 8）：
// submitAsyncAndWait 先批量提交 win 个 REQ（不 wait）填满管线，再逐个 wait 最老的
// 同时提交下一个；每线程同时有 win 个 REQ 在飞，充分利用 RDMA QP 流水线 overlap。
// 旧信号量方案因每线程 submit→wait 严格串行，每线程永远 1 个在飞（无论信号量多大），
// 8 线程最多 8 个在飞，RDMA QP 大部分时间空闲。
static size_t g_max_inflight_reqs = 8;

// 单个 NDS_REQ 的 addr_pair 物理上限（conf 键 max_addr_num_per_req）：
// 默认跟随最小 ABI 声明中的 MAX_ADDR_NUM_IPC。NdsReqWrapper 是栈上对象，越界写入
// 即栈溢出（batchPut 大批量时 SIGABRT/segfault）；maxAddrNum 由 conf 的
// req_size/batch_size 推导（如 234MB/16KB=14976），可能远超物理数组，必须 clamp。
// 厂商 SDK 更换后若数组大小变化，必须同步最小 ABI 声明并干净重编；conf 可进一步
// 调低上限，但不能用来兼容不同的 NDS_REQ 布局。
static size_t g_max_addr_num_per_req = MAX_ADDR_NUM_IPC;

// conf 推导的 maxAddrNum clamp 到物理数组上限；超过时自然拆成更多 REQ。
static inline size_t clamp_addr_num(size_t derived)
{
    return std::min(derived, g_max_addr_num_per_req);
}

// per-nsid valueSize 覆盖表（conf 键 value_size_kb_nsid<N>，N 为十进制 nsid）；
// 无覆盖的 nsid 仍使用全局 g_value_size
static std::unordered_map<uint32_t, size_t> g_value_size_by_nsid;

// key 打散开关（conf 键 key_scatter，默认 1=开启）：
// 开启后对所有 80-bit key 做 xxhash 打散再下发（见 scatter_key）。
// 注意：开关切换等于更换 key 空间，盘上旧数据全部失效，切换前需清盘或换 nsid。
static bool g_key_scatter = true;

// 已注册 MR 的范围列表（base, size），下标即 mr_id（与 nds_init 注册顺序一致）：
// 读写路径校验数据区间落在某个 MR 内，并按命中的 MR 设置 NDS_REQ.mr_id
static std::vector<std::pair<uint64_t, uint64_t>> g_mr_ranges;

// 调试开关：环境变量 NDS_WRAP_DEBUG=1 时启用调试输出
static bool g_nds_wrapper_debug_enabled = false;

// 错误打印开关（conf 键 error_print，默认 1=开启，0=关闭）：
// 断链/重连窗口内 SDK 会对每个失败 REQ 打多条 ERROR，刷屏淹没有效日志；
// 关掉后上层（LMCache adapter 的 kv_set/kv_get ret!=0 行）仍保留失败事实，
// 只是少了 wrapper 的逐 REQ 细节。排障时改回 1 即可（改 conf 生效，无需重编）。
static bool g_error_print_enabled = true;

// 辅助函数：输出带位置信息的调试信息（受环境变量控制）
static inline void nds_debug_print(const char* file, int line, const char* func, const char* fmt,
                                   ...)
{
    if (!g_nds_wrapper_debug_enabled) return;
    fprintf(stderr, "[DEBUG] %s:%d [%s] ", file, line, func);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

// 辅助函数：输出带位置信息的错误信息（受 conf error_print 控制，默认输出）
static inline void nds_error_print(const char* file, int line, const char* func, const char* fmt,
                                   ...)
{
    if (!g_error_print_enabled) return;
    fprintf(stderr, "[ERROR] %s:%d [%s] ", file, line, func);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

// 条件调试打印宏（受环境变量控制，自动添加位置信息）
#define NDS_DEBUG_PRINT(fmt, ...) nds_debug_print(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
// 错误打印宏（始终打印，自动添加位置信息）
#define NDS_ERROR_PRINT(fmt, ...) nds_error_print(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

// 80-bit key 打散（默认开启，conf key_scatter=0 关闭）：
// 对整个 80-bit key（key_low[7:0] + key_high[9:8]，含 worker_id 所在的高 16 位）
// 按小端序序列化为 10 字节后做 XXH64，输出仍为 80-bit（key_low 取第一次哈希，
// key_high 取第二次哈希的低 16 位，第二次以第一次结果为种子）。
// 目的：盘框按 key 的特定位分盘/分链时，未打散的 key 可能严重倾斜——例如上层把
// tp_rank 编进高位时高位只有几个取值，只落到少数盘。打散后全盘均匀，带宽最优。
// put/get/batchPut/batchGet/isExists 全部经本函数做同一确定性变换，读写查一致；
// 输入相同的 key 永远映射到相同的盘上 key，isExists 前缀命中语义不受影响。
// 碰撞说明：80->80 位重哈希非单射，理论存在新碰撞，但按盘框容量（几十万 key
// 量级）碰撞概率约 1e-10，可忽略；key 量到 1e9 级别时需重新评估。
static inline void scatter_key(uint64_t& key_low, uint16_t& key_high)
{
    if (!g_key_scatter) return;
    // 显式小端序列化，保证不同平台/编译器下哈希输入一致
    uint8_t buf[10];
    for (int i = 0; i < 8; ++i) { buf[i] = static_cast<uint8_t>((key_low >> (8 * i)) & 0xFF); }
    buf[8] = static_cast<uint8_t>(key_high & 0xFF);
    buf[9] = static_cast<uint8_t>((key_high >> 8) & 0xFF);
    const uint64_t h1 = XXH64(buf, sizeof(buf), 0);
    const uint64_t h2 = XXH64(buf, sizeof(buf), h1);
    key_low = h1;
    key_high = static_cast<uint16_t>(h2 & 0xFFFF);
}

// 辅助函数：检查 v1.6 SDK 返回的 CODE_SET，任一非零错误码均视为失败
static int32_t checkCodeSet(const CODE_SET& code_set, uint64_t task_id, bool read)
{
    for (size_t i = 0; i < code_set.batch_code_set.size(); ++i) {
        if (code_set.batch_code_set[i] != 0) {
            NDS_ERROR_PRINT(
                "nds_kv_wait failed, is read:%d, cid_idx:%zu, cid_err:%d, task_id:%lu\n", read, i,
                code_set.batch_code_set[i], task_id);
            return 1;
        }
    }
    for (size_t i = 0; i < code_set.key_code_set.size(); ++i) {
        if (code_set.key_code_set[i] != 0) {
            NDS_ERROR_PRINT(
                "nds_kv_wait failed, is read:%d, key_idx:%zu, key_err:%d, task_id:%lu\n", read, i,
                code_set.key_code_set[i], task_id);
            return 1;
        }
    }
    return 0;
}

// NDS_REQ 包装：v1.6 SDK 的 nds_kv_wait(task_id, total_blocks) 需要传入逻辑 block 数，
// 而不是 addr_pair 数。一个 block 可能对应多个 addr_pair，因此需要单独记录 total_blocks。
struct NdsReqWrapper {
    NDS_REQ req;
    uint16_t total_blocks;
};

// 辅助函数：为一段连续 region 生成 addr_pair。
// base_addr 为该 region 在内存中的起始地址；region_offset 为该 region 在 slot 中的全局偏移。
// key 在此统一直接打散（scatter_key）：所有写盘/读盘的 addr_pair 只出现打散后的 key，
// 原始 key 仅存在于上层日志与本函数的 debug 映射打印中。
static void appendRegionPairs(uint64_t keyLow, uint16_t keyHigh, uint8_t* base_addr,
                              size_t region_offset, size_t region_length, size_t lengthPerBatch,
                              std::vector<addr_pair>& out)
{
    if (region_length == 0) { return; }
    const uint64_t origLow = keyLow;
    const uint16_t origHigh = keyHigh;
    scatter_key(keyLow, keyHigh);
    if (g_key_scatter && (keyLow != origLow || keyHigh != origHigh)) {
        NDS_DEBUG_PRINT("key scatter: orig=(low:%llu, high:%u) -> scattered=(low:%llu, high:%u)\n",
                        (unsigned long long)origLow, origHigh, (unsigned long long)keyLow, keyHigh);
    }
    size_t numPairs = (region_length + lengthPerBatch - 1) / lengthPerBatch;
    size_t oldSize = out.size();
    out.resize(oldSize + numPairs);
    for (size_t j = 0; j < numPairs; ++j) {
        size_t offset_in_region = j * lengthPerBatch;
        addr_pair& pair = out[oldSize + j];
        pair.offset = region_offset + offset_in_region;
        pair.length = (j == numPairs - 1) ? (region_length - offset_in_region) : lengthPerBatch;
        pair.block_id.key_high = keyHigh;
        pair.block_id.key_low = keyLow;
        pair.hbm_addr =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base_addr + offset_in_region));
    }
}

// 写路径短写补齐策略（partial-slot write）：length < value_size（目标 slot 大小）时，
// 把 pair 覆盖范围扩到 [blockAddr, blockAddr+value_size)，一次单调递增写满整个
// slot，保证 is_exists 前缀命中（NDS 要求 key 写满才能命中）。
// 注意：[length, value_size) 尾部不做任何清零/填充，写的是 buffer 里的残留数据
// （don't-care，reader 按真实 length 读）；调用方必须保证 block 的 buffer 容量
// >= value_size（调用方已对 [blockAddr, blockAddr+value_size) 做 MR 边界校验）。
// 池末尾兜底：value_size 按 KB 取整必然略大于非 KB 对齐的页（如 SWA 页
// 6288.75KiB 配 6400KB），池内最后一页 contiguous 扩展必越出 MR 末尾；此时 pad 段
// 改借该 MR 头部已注册内存（同一 mr_id，内容 don't-care），末页照样写满 slot。
// 历史：曾用"自身数据平铺（tiling）"补齐，但回绕的 hbm_addr 非单调递增，在多 block
// 并发下非确定性触发 DPU "transport retry counter exceeded"，已废弃；借头部方案
// 源自 archive/padding-offset-align（0a82317，当时单 MR 全局借 pool 头部）。

// debug 辅助：写长度不是 write_batch 整数倍时提示尾 pair 为短 pair。
// 盘框若按 batch 粒度校验 pair 或按段记账，短尾 pair 可能导致 slot 不被视为写满
// （is_exists 不命中）；只在 debug 模式（NDS_WRAP_DEBUG=1）打印。
static inline void debugNoteShortTail(uint64_t blockId, uint32_t nsid, size_t length,
                                      size_t lengthPerBatch)
{
    if (lengthPerBatch != 0 && length % lengthPerBatch != 0) {
        NDS_DEBUG_PRINT(
            "write tail pair not batch-aligned: blockId=%lu, nsid=%u, length=%zu, "
            "write_batch=%zu; if is_exists misses, check box pair-granularity "
            "requirements\n",
            blockId, nsid, length, lengthPerBatch);
    }
}

// 辅助函数：查询指定 nsid 的生效 valueSize——有 per-nsid 覆盖用覆盖值，否则用全局 g_value_size
static size_t value_size_for_nsid(uint32_t nsid)
{
    auto it = g_value_size_by_nsid.find(nsid);
    if (it != g_value_size_by_nsid.end()) { return it->second; }
    return g_value_size;
}

// 静默版 MR 查找：返回 [addr, addr+length) 所在 MR 下标（即 mr_id），未命中返回 -1，
// 不打日志——供「先试扩范围、越界再走借头部兜底」这类探测性查询使用。
static int find_mr_index_nolog(uint64_t addr, size_t length)
{
    for (size_t i = 0; i < g_mr_ranges.size(); ++i) {
        const uint64_t base = g_mr_ranges[i].first;
        const uint64_t size = g_mr_ranges[i].second;
        if (size != 0 && addr >= base && addr + length <= base + size) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// 辅助函数：查找 [addr, addr+length) 落在哪个已注册 MR 内，返回其下标（即 mr_id），未命中返回 -1。
// 越界地址会让 RDMA 访问未注册/未映射内存（Work Request Flushed Error），
// 显式失败并列出全部已注册范围，比 NDS 侧模糊报错易排查得多。
static int find_mr_index(uint64_t addr, size_t length)
{
    int idx = find_mr_index_nolog(addr, length);
    if (idx >= 0) { return idx; }
    NDS_ERROR_PRINT("data range outside registered MRs: addr=%llu, length=%zu, mr_count=%zu\n",
                    (unsigned long long)addr, length, g_mr_ranges.size());
    for (size_t i = 0; i < g_mr_ranges.size(); ++i) {
        NDS_ERROR_PRINT("  mr[%zu]=[%llu,+%llu)\n", i, (unsigned long long)g_mr_ranges[i].first,
                        (unsigned long long)g_mr_ranges[i].second);
    }
    return -1;
}

// 写路径：length > value_size 报错（盘框拒绝超长写，ret:1842）；length < value_size 时
// 用自身数据平铺补齐到 value_size（debug 模式打印，平常不打印）；length == value_size 正常写。
// 读路径不补齐，只读 [offset, offset+length)，允许任意 length <= value_size。
// 历史 padding + offset 对齐完整实现见 tag archive/padding-offset-align。

// 解析 conf 数值的辅助函数：全部先按字符串取出，再按语义严格转换，
// 避免旧 %d 解析把 "2.785" 静默截断成 2（sscanf 返回成功，极难排查）。

// MB（支持小数）→ 字节。double 可精确表示 2^53 以内的整数，
// 因此上限按 2^53 字节（8 PiB）卡，保证 static_cast<size_t> 前无精度损失；
// 转换时对字节数四舍五入（如 2.785 MB = 2920284.16 B → 2920284 B）。
static bool parse_mb_to_bytes(const char* str, size_t& out)
{
    if (str == nullptr || *str == '\0') return false;
    char* end = nullptr;
    errno = 0;
    double mb = strtod(str, &end);
    if (end == str || *end != '\0' || errno == ERANGE) return false;  // 要求整串消耗
    if (!(mb > 0.0)) return false;                                    // 拒绝 0 / 负数 / NaN
    double bytes = mb * 1024.0 * 1024.0;
    if (bytes >= 9007199254740992.0) return false;  // 2^53，超出 double 精确整数范围
    out = static_cast<size_t>(bytes + 0.5);
    return true;
}

// KB（整数）→ 字节，用于需要按 KB 精确指定的场景（如 value_size_kb）
static bool parse_kb_to_bytes(const char* str, size_t& out)
{
    if (str == nullptr || *str == '\0') return false;
    char* end = nullptr;
    errno = 0;
    unsigned long long kb = strtoull(str, &end, 10);
    if (end == str || *end != '\0' || errno == ERANGE) return false;
    if (kb == 0) return false;
    if (kb > std::numeric_limits<size_t>::max() / 1024) return false;  // 防乘法溢出
    out = static_cast<size_t>(kb) * 1024;
    return true;
}

// 无单位无符号整数（如 max_exist_addr_num）
static bool parse_size_t_value(const char* str, size_t& out)
{
    if (str == nullptr || *str == '\0') return false;
    char* end = nullptr;
    errno = 0;
    unsigned long long v = strtoull(str, &end, 10);
    if (end == str || *end != '\0' || errno == ERANGE) return false;
    if (v > std::numeric_limits<size_t>::max()) return false;
    out = static_cast<size_t>(v);
    return true;
}

// 辅助函数：加载 nds_wrapper.conf（与 nds_config.conf 同目录）
static bool load_nds_wrapper_config(const char* nds_config_path)
{
    char wrapper_path[1024];
    wrapper_path[0] = '\0';

    if (nds_config_path) {
        strncpy(wrapper_path, nds_config_path, sizeof(wrapper_path) - 1);
        wrapper_path[sizeof(wrapper_path) - 1] = '\0';

        char* last_slash = strrchr(wrapper_path, '/');
        if (!last_slash) last_slash = strrchr(wrapper_path, '\\');

        if (last_slash) {
            *(last_slash + 1) = '\0';
            strncat(wrapper_path, "nds_wrapper.conf",
                    sizeof(wrapper_path) - strlen(wrapper_path) - 1);
        } else {
            strncpy(wrapper_path, "nds_wrapper.conf", sizeof(wrapper_path) - 1);
            wrapper_path[sizeof(wrapper_path) - 1] = '\0';
        }
    } else {
        strncpy(wrapper_path, "nds_wrapper.conf", sizeof(wrapper_path) - 1);
        wrapper_path[sizeof(wrapper_path) - 1] = '\0';
    }

    FILE* fp = fopen(wrapper_path, "r");
    if (!fp) {
        NDS_DEBUG_PRINT("nds_wrapper.conf not found at %s, using defaults\n", wrapper_path);
        return false;
    }

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        // 去除前导空白并跳过注释/空行
        char* p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r') p++;
        if (*p == '\n' || *p == '\0' || *p == '#') continue;

        char key[64];
        char valstr[128];
        if (sscanf(p, "%63[^=]=%127s", key, valstr) == 2) {
            // 去除 key 尾部空白
            size_t len = strlen(key);
            while (len > 0 && (key[len - 1] == ' ' || key[len - 1] == '\t')) { key[--len] = '\0'; }

            size_t parsed = 0;
            bool ok = true;
            if (strcmp(key, "read_req_size_mb") == 0) {
                ok = parse_mb_to_bytes(valstr, parsed);
                if (ok) g_length_per_read_req = parsed;
            } else if (strcmp(key, "write_req_size_mb") == 0) {
                ok = parse_mb_to_bytes(valstr, parsed);
                if (ok) g_length_per_write_req = parsed;
            } else if (strcmp(key, "read_batch_size_kb") == 0) {
                ok = parse_kb_to_bytes(valstr, parsed);
                if (ok) g_length_per_read_batch = parsed;
            } else if (strcmp(key, "write_batch_size_kb") == 0) {
                ok = parse_kb_to_bytes(valstr, parsed);
                if (ok) g_length_per_write_batch = parsed;
            } else if (strcmp(key, "value_size_mb") == 0) {
                ok = parse_mb_to_bytes(valstr, parsed);
                if (ok) g_value_size = parsed;
            } else if (strcmp(key, "value_size_kb") == 0) {
                // 需要按 KB 精确指定 value_size 时使用（优先级高于 value_size_mb，
                // 两个键同时出现时以 conf 中靠后的一行为准）
                ok = parse_kb_to_bytes(valstr, parsed);
                if (ok) g_value_size = parsed;
            } else if (strncmp(key, "value_size_kb_nsid", strlen("value_size_kb_nsid")) == 0) {
                // 按 nsid 覆盖 valueSize：value_size_kb_nsid<N> = <KB>，N 为十进制 nsid。
                // 后缀非法（空/非数字/含尾随字符/超 uint32）或值非法均视为解析失败，
                // 与其他键一样保留前值并显式报错
                const char* nsid_str = key + strlen("value_size_kb_nsid");
                char* end = nullptr;
                errno = 0;
                unsigned long long nsid_val = strtoull(nsid_str, &end, 10);
                if (end == nsid_str || *end != '\0' || errno == ERANGE ||
                    nsid_val > std::numeric_limits<uint32_t>::max()) {
                    ok = false;
                } else {
                    ok = parse_kb_to_bytes(valstr, parsed);
                    if (ok) g_value_size_by_nsid[static_cast<uint32_t>(nsid_val)] = parsed;
                }
            } else if (strcmp(key, "max_exist_addr_num") == 0) {
                ok = parse_size_t_value(valstr, parsed);
                if (ok) g_max_exist_addr_num = parsed;
            } else if (strcmp(key, "key_scatter") == 0) {
                // key 打散开关：1=开启（默认），0=关闭。只接受 0/1。
                // 切换等于更换 key 空间（盘上旧数据失效），切换前需清盘或换 nsid。
                ok = parse_size_t_value(valstr, parsed);
                if (ok && parsed <= 1) {
                    g_key_scatter = (parsed != 0);
                } else {
                    ok = false;
                }
            } else if (strcmp(key, "error_print") == 0) {
                // wrapper [ERROR] 打印开关：1=开启（默认），0=关闭。只接受 0/1。
                // 断链/重连窗口内 SDK 对每个失败 REQ 打多条 ERROR 会刷屏淹没
                // 有效日志；关闭后失败事实仍由上层（LMCache adapter 的
                // kv_set/kv_get ret!=0 行）体现，排障时改回 1 即可。
                ok = parse_size_t_value(valstr, parsed);
                if (ok && parsed <= 1) {
                    g_error_print_enabled = (parsed != 0);
                } else {
                    ok = false;
                }
            } else if (strcmp(key, "max_inflight_reqs") == 0) {
                // 每线程滑动窗口大小（默认 8，拒绝 0）：submitAsyncAndWait 先批量
                // 提交 win 个 REQ 填满管线，再滑动 wait+submit，每线程同时 win 个
                // REQ 在飞。调参：从 8 起步（8 线程 × 8 = 64 并发），带宽不够
                // 再加到 16。总并发 ≈ 线程数 × win，需控制在 SDK 安全上限内。
                ok = parse_size_t_value(valstr, parsed);
                if (ok && parsed > 0 && parsed <= 65536) {
                    g_max_inflight_reqs = parsed;
                } else {
                    ok = false;
                }
            } else if (strcmp(key, "max_addr_num_per_req") == 0) {
                // 单 REQ addr_pair 物理上限，默认跟随本文件的 MAX_ADDR_NUM_IPC。
                // conf 只能调低，不能超过定长数组容量；SDK 数组布局变化时必须
                // 同步 ABI 声明并干净重编。拒绝 0 和越界值。
                ok = parse_size_t_value(valstr, parsed);
                if (ok && parsed > 0 && parsed <= MAX_ADDR_NUM_IPC) {
                    g_max_addr_num_per_req = parsed;
                } else {
                    ok = false;
                }
            }
            // put_threads / put_core_start / get_threads 已废弃：改为滑动窗口
            // 模型（max_inflight_reqs），旧 conf 含这些键时静默忽略

            if (!ok) {
                // 解析失败：保留默认值/前值，显式报错避免静默截断类问题
                NDS_ERROR_PRINT("invalid value for '%s': '%s', keeping previous setting\n", key,
                                valstr);
            }
        }
    }
    fclose(fp);

    // value_size 与 write_batch 对齐性检查（常驻警告，每次 init 打一次）：
    // 实测盘框对非 write_batch 倍数长度的 slot，短尾 pair 可能未被计入"写满"判定，
    // 表现为 batchPut 成功但 is_exists 永不命中（如 5616KB = 43.875 x 128KB）。
    // wrapper 的 pair 覆盖是完整的（尾 pair 为短 pair），不丢数据；但若盘框按
    // batch 粒度记账，唯一稳妥做法是 value_size 按 write_batch_size 对齐。
    if (g_length_per_write_batch != 0 && g_value_size % g_length_per_write_batch != 0) {
        NDS_ERROR_PRINT(
            "WARNING: value_size=%zu is not a multiple of write_batch=%zu; "
            "short tail pair may not count toward a full slot on the box "
            "(is_exists may never hit). Recommend aligning value_size to a "
            "multiple of write_batch_size_kb\n",
            g_value_size, g_length_per_write_batch);
    }
    for (const auto& entry : g_value_size_by_nsid) {
        if (g_length_per_write_batch != 0 && entry.second % g_length_per_write_batch != 0) {
            NDS_ERROR_PRINT(
                "WARNING: nsid=%u value_size=%zu is not a multiple of write_batch=%zu; "
                "is_exists may never hit. Recommend aligning to a multiple of "
                "write_batch_size_kb\n",
                entry.first, entry.second, g_length_per_write_batch);
        }
    }

    // value_size 与 read_batch 对齐性检查（常驻警告）：读路径把尾 pair 补齐到
    // read_batch 整数倍，clamp 到 slot 末尾时的对齐依赖 value_size 是 read_batch
    // 的整数倍；否则读请求仍带非对齐尾 pair，盘框返回 1842 (NOTFULLYCOMPLETED)。
    if (g_length_per_read_batch != 0 && g_value_size % g_length_per_read_batch != 0) {
        NDS_ERROR_PRINT(
            "WARNING: value_size=%zu is not a multiple of read_batch=%zu; "
            "extended read tail pair stays unaligned and the box may return "
            "1842. Recommend aligning value_size to a multiple of "
            "read_batch_size_kb\n",
            g_value_size, g_length_per_read_batch);
    }
    for (const auto& entry : g_value_size_by_nsid) {
        if (g_length_per_read_batch != 0 && entry.second % g_length_per_read_batch != 0) {
            NDS_ERROR_PRINT(
                "WARNING: nsid=%u value_size=%zu is not a multiple of read_batch=%zu; "
                "extended read may return 1842. Recommend aligning to a multiple of "
                "read_batch_size_kb\n",
                entry.first, entry.second, g_length_per_read_batch);
        }
    }

    // addr_pair 物理上限检查（常驻警告）：conf 推导的 maxAddrNum（req/batch）超过
    // SDK addr_pair_list 数组大小时，越界写入栈上 NdsReqWrapper 即 SIGABRT/segfault
    // （实测 234MB/16KB=14976 >> 4096，batchPut ≥12 blocks 崩溃）。装箱路径已按
    // max_addr_num_per_req clamp（功能正确，REQ 数变多），但建议调 req/batch 消除
    // 额外拆分。
    const size_t derivedWriteAddrNum =
        g_length_per_write_batch != 0 ? g_length_per_write_req / g_length_per_write_batch : 0;
    const size_t derivedReadAddrNum =
        g_length_per_read_batch != 0 ? g_length_per_read_req / g_length_per_read_batch : 0;
    if (derivedWriteAddrNum > g_max_addr_num_per_req ||
        derivedReadAddrNum > g_max_addr_num_per_req) {
        NDS_ERROR_PRINT(
            "WARNING: derived maxAddrNum (write=%zu, read=%zu) exceeds "
            "max_addr_num_per_req=%zu (SDK addr_pair_list capacity); "
            "clamping per-REQ pair count (safe, but more REQs). Recommend "
            "lowering *_req_size_mb or raising *_batch_size_kb so "
            "req_size/batch_size <= %zu\n",
            derivedWriteAddrNum, derivedReadAddrNum, g_max_addr_num_per_req,
            g_max_addr_num_per_req);
    }

    NDS_DEBUG_PRINT(
        "nds_wrapper config: read_req=%zuM, write_req=%zuM, read_batch=%zuK, write_batch=%zuK, "
        "value_size=%zu bytes (%.3f MB), max_exist=%zu, key_scatter=%d, max_inflight_reqs=%zu, "
        "max_addr_num_per_req=%zu, error_print=%d\n",
        g_length_per_read_req / (1024 * 1024), g_length_per_write_req / (1024 * 1024),
        g_length_per_read_batch / 1024, g_length_per_write_batch / 1024, g_value_size,
        g_value_size / 1048576.0, g_max_exist_addr_num, g_key_scatter ? 1 : 0, g_max_inflight_reqs,
        g_max_addr_num_per_req, g_error_print_enabled ? 1 : 0);
    // per-nsid valueSize 覆盖逐项打印（unordered_map 顺序不定，仅调试用）
    for (const auto& entry : g_value_size_by_nsid) {
        NDS_DEBUG_PRINT(
            "nds_wrapper config: nsid=%u value_size=%zu bytes (%.3f MB) [per-nsid override]\n",
            entry.first, entry.second, entry.second / 1048576.0);
    }
    return true;
}

// task_id 全局唯一生成：原子计数器，初值混入时间戳，避免进程重启后与上一轮 task 冲突。
// 旧实现用 thread_local mt19937_64（时间 + random_device 做种子）：多个线程同时初始化
// thread_local RNG 时可能拿到相同种子，而 mt19937_64 是确定性的，相同种子产生相同序列；
// 两个 NDS_REQ 拿到相同 task_id 时，后发的 nds_kv_put 会覆盖前者的 task 状态，
// 导致前者的 nds_kv_wait 永久 hang。isExists 旧用的全局 rng 在多线程下还有数据竞争。
static std::atomic<uint64_t> g_task_id_counter{
    static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};

static inline uint64_t next_task_id()
{
    return g_task_id_counter.fetch_add(1, std::memory_order_relaxed);
}

// 批量读写（batchGet/batchPut）改为滑动窗口模型（见 submitAsyncAndWait）：
// 调用线程内先批量提交 win 个 REQ 填充管线，再逐个 wait 最老的 + submit 下一个；
// 每线程同时有 g_max_inflight_reqs 个 REQ 在飞（conf 键，默认 8）。
// 旧两阶段模型（全部下发再全部 wait）已被替换，因其多线程并发时一次性压入大量 REQ
// （8 线程 × 128 REQ/批 = 1024 在飞），超出 NDS SDK QP / 在途 task 容量致 hang。
// 读/写线程池已全部移除，conf 中的 get_threads / put_threads / put_core_start 键
// 随之废弃（不再读取，保留键仅为兼容旧 conf 文件）。

// init / initMulti 共享的初始化前导：加载 nds config、NDS_WRAP_DEBUG 调试开关、wrapper conf
static void initPreamble(const char* path_nds_config)
{
    const char* config_path = path_nds_config;
    if (config_path == nullptr || strcmp(config_path, "") == 0) { config_path = "nds_config.conf"; }

    nds_kv_get_nds_config(config_path);

    const char* env = std::getenv("NDS_WRAP_DEBUG");
    g_nds_wrapper_debug_enabled = (env != nullptr && std::atoi(env) != 0);

    load_nds_wrapper_config(config_path);
}

int32_t NDS::initMulti(std::vector<void*> memAddrs, std::vector<uint64_t> lengths,
                       const char* path_nds_config)
{
    if (memAddrs.empty()) {
        NDS_ERROR_PRINT("initMulti: memAddrs is empty\n");
        return -1;
    }
    if (memAddrs.size() != lengths.size()) {
        NDS_ERROR_PRINT("initMulti: memAddrs/lengths size mismatch: addrs=%zu, lengths=%zu\n",
                        memAddrs.size(), lengths.size());
        return -1;
    }
    for (size_t i = 0; i < lengths.size(); ++i) {
        if (lengths[i] == 0) {
            NDS_ERROR_PRINT("initMulti: zero length at index %zu\n", i);
            return -1;
        }
    }
    // 预校验基地址页对齐：SDK 注册 MR 要求 4096 对齐，但对部分 MR 注册失败
    // 不会通过 nds_init 返回值反映（实测出现 "Index 1: Address ... NOT page
    // aligned / Failed to register mr: Bad address" 后 nds_init 仍返回 0 的
    // 半注册状态，后续零拷贝 IO 打到未注册区域会段错误）。这里提前 fail-fast，
    // 让上层（Python init_once_multi）回退到只注册 anchor buffer（staging 兜底），
    // 而不是带着残缺的 MR 表继续跑。
    for (size_t i = 0; i < memAddrs.size(); ++i) {
        auto addrVal = reinterpret_cast<uintptr_t>(memAddrs[i]);
        if (addrVal % 4096 != 0) {
            NDS_ERROR_PRINT(
                "initMulti: address at index %zu (0x%llx) is not 4096-aligned; "
                "refusing to register any MR (partial registration is unsafe)\n",
                i, (unsigned long long)static_cast<uint64_t>(addrVal));
            return -1;
        }
    }

    initPreamble(path_nds_config);

    // 注册全部 MR。SDK 端访问语义（NDS_client.c）：对 *kv_cache 解引用一次后当
    // 【连续 kv_mr[] 数组】访问（mrs[i].addr），而不是访问 kv_mr* 指针数组。
    // 因此必须传「指向连续 kv_mr 数组的指针」。kv_mr* 指针数组会让 SDK 的
    // mrs[i]（i≥1）越过第一个指针读到指针数组/堆元数据当 addr 解释，报
    // "Index N: Address 0x.. is NOT page aligned"（实测垃圾值如 0x20、
    // 0x65686361635f65——ASCII 字符串内存）。单 MR 因只读 mrs[0] 碰巧不触发。
    // 历史：d5aa6ba 曾把 unique_ptr 数组改为裸 kv_mr* 指针数组，消除了
    // deleter padding 误读，但指针数组本身仍不符合 SDK 的连续数组语义，
    // 多 MR 下依旧读到分散堆对象——369ada8 按内部验证过的修复改为真正的
    // 连续 kv_mr 数组。
    //
    // 生命周期：断链重连时 NDS_client 会【通过 nds_init 保存下来的指针重读
    // MR 表重新注册】，而非 init 时一次性深拷贝。此前 mr_array 是 initMulti
    // 栈上局部 vector，nds_init 返回后即析构，重连重注册读到的是已释放/
    // 被复用的堆内存，报 "Index 0: Address 0x<垃圾值> is NOT page aligned"
    // （实测 0x481775c08548595a，x86-64 非规范地址，断链前同一 buffer 注册
    // 正常、conn 0~7 重连全部坏在同一垃圾值——典型的 dangling pointer 特征）。
    // 因此 mr_array 必须进程级存活：static 堆对象、有意不释放。lazy retry
    // 二次 init 时原地 clear+填充（mrNum 不超过首次容量时 data() 不变），
    // SDK 保存的指针始终指向有效 MR 表。
    //
    // 外层指针同理：nds_init 的形参是 void**（即 &mr_base），若 SDK 存的是
    // 这个外层地址、重连时才解引用 *kv_cache，那么栈上局部 mr_base 本身也是
    // dangling（1s 断连测试能过可能只是栈槽位碰巧残留旧值）。内层数组和外层
    // 指针都必须进程级存活，两种 SDK 实现都安全。
    const size_t mrNum = memAddrs.size();
    static std::vector<kv_mr>* g_mr_array = new std::vector<kv_mr>();  // 有意泄漏，进程生命周期
    static kv_mr* g_mr_base = nullptr;  // 外层指针同样进程级存活（nds_init 存的是它的地址）
    g_mr_array->clear();
    g_mr_array->reserve(mrNum);
    for (size_t i = 0; i < mrNum; ++i) {
        auto addrVal = reinterpret_cast<uintptr_t>(memAddrs[i]);
        // kv_mr 是聚合类型（只有默认构造），必须走 brace-init，不能 emplace_back
        g_mr_array->push_back(kv_mr{static_cast<uint64_t>(addrVal), lengths[i]});
    }
    g_mr_base = g_mr_array->data();
    auto ret = nds_init(reinterpret_cast<void**>(&g_mr_base), static_cast<int>(mrNum));
    if (ret != 0) {
        NDS_ERROR_PRINT("Failed to register dram pool\n");
        return -1;
    }
    // 记录全部 MR 范围（下标即 mr_id），供读写路径校验数据区间并设置 req.mr_id
    g_mr_ranges.clear();
    g_mr_ranges.reserve(mrNum);
    for (size_t i = 0; i < mrNum; ++i) {
        auto addrVal = reinterpret_cast<uintptr_t>(memAddrs[i]);
        g_mr_ranges.emplace_back(static_cast<uint64_t>(addrVal), lengths[i]);
        NDS_DEBUG_PRINT("registered mr[%zu]=[%llu,+%llu)\n", i,
                        (unsigned long long)static_cast<uint64_t>(addrVal),
                        (unsigned long long)lengths[i]);
    }
    return 0;
}

int32_t NDS::init(void* memAddr, uint64_t length, const char* path_nds_config)
{
    // 单 buffer 即单元素 initMulti（与原实现行为一致：成功返回 0，失败打印并返回 -1）
    return NDS::initMulti({memAddr}, {length}, path_nds_config);
}

int32_t NDS::isExists(std::vector<uint64_t> blockIdLows, std::vector<uint16_t> blockIdHighs,
                      uint32_t nsid)
{
    if (blockIdLows.size() != blockIdHighs.size()) {
        NDS_ERROR_PRINT("isExists: blockIdLows/blockIdHighs size mismatch: %zu vs %zu\n",
                        blockIdLows.size(), blockIdHighs.size());
        return -1;
    }
    // 同样 clamp 到 addr_pair_list 物理上限：exist 查询也写同一个定长数组
    const size_t maxExistAddrNum = clamp_addr_num(g_max_exist_addr_num);
    auto reqNum = (blockIdLows.size() + maxExistAddrNum - 1) / maxExistAddrNum;
    std::vector<NDS_REQ> reqs(reqNum);
    auto idx = 0;
    for (uint64_t i = 0; i < reqNum; ++i) {
        auto addrNum =
            i < (reqNum - 1) ? maxExistAddrNum : (blockIdLows.size() - i * maxExistAddrNum);
        reqs[i].task_id = next_task_id();
        reqs[i].addr_num = addrNum;
        reqs[i].mr_id = 0;
        reqs[i].nsid = nsid;
        for (uint64_t j = 0; j < addrNum; ++j) {
            // 与读写路径同一变换：查询必须用打散后的 key，否则永远 miss
            uint64_t keyLow = blockIdLows[idx];
            uint16_t keyHigh = blockIdHighs[idx];
            ++idx;
            scatter_key(keyLow, keyHigh);
            reqs[i].addr_pair_list[j].block_id.key_high = keyHigh;
            reqs[i].addr_pair_list[j].block_id.key_low = keyLow;
        }
    }

    int32_t res = 0;
    uint64_t cur;
    for (auto& req : reqs) {
        cur = nds_kv_exist(&req);
        res += static_cast<int32_t>(cur);
        if (cur < req.addr_num) { break; }
    }
    return res;
}

std::vector<NdsReqWrapper> prepareNdsReqsBatch(const std::vector<uint64_t>& blockIds,
                                               const std::vector<uint16_t>& blockIdHighs,
                                               const std::vector<uint8_t*>& blockAddrs,
                                               const std::vector<size_t>& offsets,
                                               const std::vector<size_t>& lengths,
                                               const std::vector<uint32_t>& nsids, bool read)
{
    if (blockIds.empty()) { return {}; }
    const size_t expectedSize = blockIds.size();
    if (blockIdHighs.size() != expectedSize || blockAddrs.size() != expectedSize ||
        offsets.size() != expectedSize || lengths.size() != expectedSize ||
        nsids.size() != expectedSize) {
        NDS_ERROR_PRINT(
            "input vector size mismatch: blockIds=%zu, blockIdHighs=%zu, blockAddrs=%zu, "
            "offsets=%zu, lengths=%zu, nsids=%zu\n",
            blockIds.size(), blockIdHighs.size(), blockAddrs.size(), offsets.size(), lengths.size(),
            nsids.size());
        return {};
    }
    const size_t lengthPerBatch = read ? g_length_per_read_batch : g_length_per_write_batch;
    // clamp 到 SDK addr_pair_list 物理上限（max_addr_num_per_req，默认 4096），
    // 防止 conf 推导值（req/batch）超过定长数组导致栈溢出；超过时自然拆成更多 REQ
    const size_t maxAddrNum =
        clamp_addr_num(read ? (g_length_per_read_req / g_length_per_read_batch)
                            : (g_length_per_write_req / g_length_per_write_batch));
    std::vector<std::vector<addr_pair>> blockPairsCache;
    std::vector<int> mrs(expectedSize, -1);  // 各 block 数据区间命中的 MR 下标（即 mr_id）
    size_t totalPairs = 0;
    for (size_t i = 0; i < expectedSize; ++i) {
        if (lengths[i] == 0) {
            NDS_ERROR_PRINT("invalid io size: blockId=%lu, length=%zu\n", blockIds[i], lengths[i]);
            return {};
        }
        const size_t expectValueSize = value_size_for_nsid(nsids[i]);
        mrs[i] = find_mr_index(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(blockAddrs[i])),
                               lengths[i]);
        if (mrs[i] < 0) {
            NDS_ERROR_PRINT("block addr not in any registered MR: blockId=%lu\n", blockIds[i]);
            return {};
        }

        blockPairsCache.emplace_back();
        auto& currentPairs = blockPairsCache.back();
        if (read) {
            // 读路径尾 pair 补齐到 read_batch 整数倍：盘框要求 RDMA READ 的尾 pair
            // 长度按 read_batch 对齐，非对齐碎片返回 1842 (NOTFULLYCOMPLETED)。
            // 只向上取整到 batch 粒度（放大量 < 1 个 batch），不读到 slot 末尾，
            // 避免小 length 大 slot 时的严重读放大。多读的 [length, aligned) 落在
            // 调用方 block buffer 内（容量 >= value_size，与 partial-slot write
            // 同一前提），调用方只取前 length 字节。
            if (offsets[i] + lengths[i] > expectValueSize) {
                NDS_ERROR_PRINT(
                    "offset overflow: blockId=%lu, offset=%zu, length=%zu, value_size=%zu\n",
                    blockIds[i], offsets[i], lengths[i], expectValueSize);
                return {};
            }
            size_t readLen = (lengths[i] + lengthPerBatch - 1) / lengthPerBatch * lengthPerBatch;
            const size_t slotRemain = expectValueSize - offsets[i];
            if (readLen > slotRemain) {
                // clamp 到 slot 末尾；此时尾 pair 对齐依赖
                // (value_size - offset) % read_batch == 0（init 有常驻 WARNING 检查
                // value_size % read_batch；offset 非 batch 对齐时无法保证，属调用方约束）
                readLen = slotRemain;
                if (readLen < lengths[i]) {
                    // clamp 后短于请求长度 = slotRemain 非 batch 对齐（offset 未按
                    // batch 对齐），读全必然 1842；显式报错而非静默截断数据
                    NDS_ERROR_PRINT(
                        "aligned read: blockId=%lu, slot remainder %zu not batch-aligned "
                        "(offset=%zu, length=%zu, value_size=%zu, read_batch=%zu); "
                        "align read offset to read_batch_size_kb\n",
                        blockIds[i], slotRemain, offsets[i], lengths[i], expectValueSize,
                        lengthPerBatch);
                    return {};
                }
            }
            // 扩范围后必须单独校验整段落在 MR 内（mrs[i] 只校验了 [addr, addr+length)）
            if (readLen > lengths[i] &&
                find_mr_index(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(blockAddrs[i])),
                              readLen) < 0) {
                NDS_ERROR_PRINT("aligned read: blockId=%lu, addr=%p, readLen=%zu outside MR\n",
                                blockIds[i], (void*)blockAddrs[i], readLen);
                return {};
            }
            if (readLen != lengths[i]) {
                NDS_DEBUG_PRINT("aligned read: blockId=%lu, nsid=%u, dataLen=%zu, alignedLen=%zu\n",
                                blockIds[i], nsids[i], lengths[i], readLen);
            }
            currentPairs.reserve((readLen + lengthPerBatch - 1) / lengthPerBatch);
            appendRegionPairs(blockIds[i], blockIdHighs[i], blockAddrs[i], offsets[i], readLen,
                              lengthPerBatch, currentPairs);
        } else if (lengths[i] > expectValueSize) {
            // 超长写：盘框拒绝（ret:1842），提前拦截并给出建议配置值
            size_t suggest_kb = (lengths[i] + 1023) / 1024;
            const char* roundNote =
                (lengths[i] % 1024 == 0) ? "" : "（length 非 KB 对齐，已向上取整）";
            if (g_value_size_by_nsid.count(nsids[i]) != 0) {
                NDS_ERROR_PRINT(
                    "length > value_size: blockId=%lu, nsid=%u, length=%zu, value_size=%zu; "
                    "建议将盘框 nsid=%u 的 valueSize 与 value_size_kb_nsid%u 配置为 %zu%s\n",
                    blockIds[i], nsids[i], lengths[i], expectValueSize, nsids[i], nsids[i],
                    suggest_kb, roundNote);
            } else {
                NDS_ERROR_PRINT(
                    "length > value_size: blockId=%lu, nsid=%u, length=%zu, value_size=%zu; "
                    "建议将盘框 nsid=%u 的 valueSize 与 value_size_kb 配置为 %zu，"
                    "或新增 value_size_kb_nsid%u=%zu%s\n",
                    blockIds[i], nsids[i], lengths[i], expectValueSize, nsids[i], suggest_kb,
                    nsids[i], suggest_kb, roundNote);
            }
            return {};
        } else if (lengths[i] < expectValueSize) {
            // 短写补齐：pair 覆盖范围直接扩到 [blockAddr, blockAddr+value_size)，
            // 单调递增一次写满整个 slot（尾部 [length, value_size) 为 buffer 残留
            // 数据，don't-care，reader 按真实 length 读，不做任何清零动作）。
            // 策略演进见 appendRegionPairs 上方的注释。
            if (offsets[i] != 0) {
                NDS_ERROR_PRINT(
                    "non-zero write offset not allowed for padded write: blockId=%lu, offset=%zu\n",
                    blockIds[i], offsets[i]);
                return {};
            }
            // 补齐后 RDMA READ 范围扩到 value_size。优先 contiguous 扩展（pair 偏移
            // 单调递增，性能最佳）；但当 block 落在池末尾、扩范围越出所属 MR 时
            // （value_size 按 KB 取整必然略大于非 KB 对齐的页，如 SWA 页
            // 6288.75KiB 配 6400KB，最后一页必越界），pad 段改借该 MR 头部的
            // 已注册内存（同一 mr_id，内容 don't-care），让末页也能写满 slot、
            // is_exists 语义不受损。借头部移植自 archive/padding-offset-align
            // （0a82317），当年正是为解决同一问题引入。
            uint64_t blockAddrVal =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(blockAddrs[i]));
            if (find_mr_index_nolog(blockAddrVal, expectValueSize) >= 0) {
                NDS_DEBUG_PRINT(
                    "padded write: blockId=%lu, nsid=%u, dataLen=%zu, writing full slot=%zu\n",
                    blockIds[i], nsids[i], lengths[i], expectValueSize);
                currentPairs.reserve((expectValueSize + lengthPerBatch - 1) / lengthPerBatch);
                appendRegionPairs(blockIds[i], blockIdHighs[i], blockAddrs[i], 0, expectValueSize,
                                  lengthPerBatch, currentPairs);
            } else {
                // mrs[i] 已校验 [addr, addr+length) 落在该 MR 内，直接借其头部
                const uint64_t mrBase = g_mr_ranges[mrs[i]].first;
                const uint64_t mrSize = g_mr_ranges[mrs[i]].second;
                if (mrSize < expectValueSize) {
                    NDS_ERROR_PRINT(
                        "padded write: blockId=%lu, addr=%p, value_size=%zu exceeds "
                        "its MR [%llu,+%llu) and MR is smaller than value_size; "
                        "cannot borrow MR head for padding\n",
                        blockIds[i], (void*)blockAddrs[i], expectValueSize,
                        (unsigned long long)mrBase, (unsigned long long)mrSize);
                    return {};
                }
                // 注意：pad 段在 lengths[i] 处引入新的 pair 边界，length 非
                // write_batch 对齐时 pad 段首 pair 为短 pair（与旧实现语义一致，
                // test_padding 实测 is_exists 可命中）。
                NDS_DEBUG_PRINT(
                    "padded write borrow MR head: blockId=%lu, nsid=%u, dataLen=%zu, "
                    "padLen=%zu borrowed from mr[%d] base\n",
                    blockIds[i], nsids[i], lengths[i], expectValueSize - lengths[i], mrs[i]);
                currentPairs.reserve((expectValueSize + lengthPerBatch - 1) / lengthPerBatch + 1);
                appendRegionPairs(blockIds[i], blockIdHighs[i], blockAddrs[i], 0, lengths[i],
                                  lengthPerBatch, currentPairs);
                uint8_t* padSrc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(mrBase));
                appendRegionPairs(blockIds[i], blockIdHighs[i], padSrc, lengths[i],
                                  expectValueSize - lengths[i], lengthPerBatch, currentPairs);
            }
        } else {
            debugNoteShortTail(blockIds[i], nsids[i], lengths[i], lengthPerBatch);
            currentPairs.reserve((lengths[i] + lengthPerBatch - 1) / lengthPerBatch);
            appendRegionPairs(blockIds[i], blockIdHighs[i], blockAddrs[i], offsets[i], lengths[i],
                              lengthPerBatch, currentPairs);
        }
        totalPairs += currentPairs.size();
    }

    std::vector<NdsReqWrapper> reqs;
    std::vector<size_t> nextPairIdx(blockPairsCache.size(), 0);
    size_t blockIdx = 0;

    NDS_DEBUG_PRINT(
        "****** prepareNdsReqsBatch, is_read: %d, lengthPerBatch: %zu, maxAddrNum: %zu, "
        "blockCount: %zu, totalPairs: %zu\n",
        read, lengthPerBatch, maxAddrNum, blockPairsCache.size(), totalPairs);

    // 按 block 轮转混装：同一 (nsid, mr_id) 连续段内，每个 REQ 以 pair 为粒度轮流取各 block
    // 的下一个 pair，直到装满 maxAddrNum。单个 REQ 因此横跨多个 key，经哈希环自然
    // 铺到多条链（对齐 testtool keyswitch=on 的形态，单 REQ 即多链）；nsid 或 mr_id
    // 变化即截断另起新 REQ（一个 NDS_REQ 只有单一 nsid/mr_id 字段）。单 block 超
    // maxAddrNum 的拆分场景由同一逻辑自然覆盖（段内只剩一个 block 时退化为顺序切分）。
    // 注意：一个 block 的 pair 可能横跨多个 REQ，total_blocks 统计的是"本 REQ 覆盖
    // 到的 distinct block 数"（与 nds_kv_wait 的 key 粒度一致），因此用 reqSeq 代际
    // 戳标记，跨 REQ 复用时重新计数。
    const size_t totalBlockCount = blockPairsCache.size();
    std::vector<size_t> countedGen(totalBlockCount, SIZE_MAX);  // block 被哪个 REQ 计过数
    size_t reqSeq = 0;
    while (blockIdx < totalBlockCount) {
        // 跳过已消费完的 block
        if (nextPairIdx[blockIdx] >= blockPairsCache[blockIdx].size()) {
            ++blockIdx;
            continue;
        }

        NdsReqWrapper wrapper;
        wrapper.req = NDS_REQ{};
        wrapper.total_blocks = 0;
        size_t filledPairs = 0;
        // 当前 REQ 的 (nsid, mr_id)：由段内首个未消费完的 block 决定
        const uint32_t reqNsid = nsids[blockIdx];
        const int reqMr = mrs[blockIdx];

        // 同 (nsid, mr_id) 连续段 [blockIdx, runEnd)
        size_t runEnd = blockIdx;
        while (runEnd < totalBlockCount && nsids[runEnd] == reqNsid && mrs[runEnd] == reqMr) {
            ++runEnd;
        }

        // 轮转取 pair：每轮从段内每个未消费完的 block 各取一个，直到 REQ 装满
        bool anyLeft = true;
        while (filledPairs < maxAddrNum && anyLeft) {
            anyLeft = false;
            for (size_t b = blockIdx; b < runEnd && filledPairs < maxAddrNum; ++b) {
                if (nextPairIdx[b] < blockPairsCache[b].size()) {
                    if (countedGen[b] != reqSeq) {
                        countedGen[b] = reqSeq;
                        wrapper.total_blocks++;  // 本 REQ 首次覆盖该 block
                    }
                    wrapper.req.addr_pair_list[filledPairs++] =
                        blockPairsCache[b][nextPairIdx[b]++];
                    anyLeft = true;
                }
            }
        }

        wrapper.req.addr_num = static_cast<uint16_t>(filledPairs);
        wrapper.req.task_id = next_task_id();
        wrapper.req.mr_id = reqMr;
        wrapper.req.nsid = reqNsid;
        reqs.push_back(wrapper);
        ++reqSeq;

        // blockIdx 前进到段内第一个未消费完的 block（可能仍在段内，下一个 REQ 继续）
        while (blockIdx < runEnd && nextPairIdx[blockIdx] >= blockPairsCache[blockIdx].size()) {
            ++blockIdx;
        }
    }
    NDS_DEBUG_PRINT("****** prepareNdsReqsBatch done, reqCount: %zu\n", reqs.size());
    return reqs;
}

std::vector<NdsReqWrapper> prepareNdsReqs(uint64_t blockId, uint16_t blockIdHigh,
                                          uint8_t* blockAddr, size_t offset, size_t length,
                                          uint32_t nsid, bool read)
{
    const size_t lengthPerBatch = read ? g_length_per_read_batch : g_length_per_write_batch;
    // clamp 到 SDK addr_pair_list 物理上限（max_addr_num_per_req，默认 4096），
    // 防止 conf 推导值（req/batch）超过定长数组导致栈溢出；超过时自然拆成更多 REQ
    const size_t maxAddrNum =
        clamp_addr_num(read ? (g_length_per_read_req / g_length_per_read_batch)
                            : (g_length_per_write_req / g_length_per_write_batch));

    if (length == 0) {
        NDS_ERROR_PRINT("invalid io size: length is zero\n");
        return {};
    }
    const size_t expectValueSize = value_size_for_nsid(nsid);
    const int mr =
        find_mr_index(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(blockAddr)), length);
    if (mr < 0) {
        NDS_ERROR_PRINT("block addr not in any registered MR: blockId=%lu\n", blockId);
        return {};
    }

    std::vector<addr_pair> pairs;
    if (read) {
        // 读路径尾 pair 补齐到 read_batch 整数倍：盘框要求 RDMA READ 的尾 pair
        // 长度按 read_batch 对齐，非对齐碎片返回 1842 (NOTFULLYCOMPLETED)。
        // 只向上取整到 batch 粒度（放大量 < 1 个 batch），不读到 slot 末尾，
        // 避免小 length 大 slot 时的严重读放大。多读的 [length, aligned) 落在
        // 调用方 block buffer 内（容量 >= value_size，与 partial-slot write
        // 同一前提），调用方只取前 length 字节。
        if (offset + length > expectValueSize) {
            NDS_ERROR_PRINT(
                "offset overflow: blockId=%lu, offset=%zu, length=%zu, value_size=%zu\n", blockId,
                offset, length, expectValueSize);
            return {};
        }
        size_t readLen = (length + lengthPerBatch - 1) / lengthPerBatch * lengthPerBatch;
        const size_t slotRemain = expectValueSize - offset;
        if (readLen > slotRemain) {
            // clamp 到 slot 末尾；此时尾 pair 对齐依赖
            // (value_size - offset) % read_batch == 0（init 有常驻 WARNING 检查
            // value_size % read_batch；offset 非 batch 对齐时无法保证，属调用方约束）
            readLen = slotRemain;
            if (readLen < length) {
                // clamp 后短于请求长度 = slotRemain 非 batch 对齐（offset 未按
                // batch 对齐），读全必然 1842；显式报错而非静默截断数据
                NDS_ERROR_PRINT(
                    "aligned read: blockId=%lu, slot remainder %zu not batch-aligned "
                    "(offset=%zu, length=%zu, value_size=%zu, read_batch=%zu); "
                    "align read offset to read_batch_size_kb\n",
                    blockId, slotRemain, offset, length, expectValueSize, lengthPerBatch);
                return {};
            }
        }
        // 扩范围后必须单独校验整段落在 MR 内（mr 只校验了 [addr, addr+length)）
        if (readLen > length &&
            find_mr_index(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(blockAddr)), readLen) <
                0) {
            NDS_ERROR_PRINT("aligned read: blockId=%lu, addr=%p, readLen=%zu outside MR\n", blockId,
                            (void*)blockAddr, readLen);
            return {};
        }
        if (readLen != length) {
            NDS_DEBUG_PRINT("aligned read: blockId=%lu, nsid=%u, dataLen=%zu, alignedLen=%zu\n",
                            blockId, nsid, length, readLen);
        }
        pairs.reserve((readLen + lengthPerBatch - 1) / lengthPerBatch);
        appendRegionPairs(blockId, blockIdHigh, blockAddr, offset, readLen, lengthPerBatch, pairs);
    } else if (length > expectValueSize) {
        // 超长写：盘框拒绝（ret:1842），提前拦截并给出建议配置值
        size_t suggest_kb = (length + 1023) / 1024;
        const char* roundNote = (length % 1024 == 0) ? "" : "（length 非 KB 对齐，已向上取整）";
        if (g_value_size_by_nsid.count(nsid) != 0) {
            NDS_ERROR_PRINT(
                "length > value_size: blockId=%lu, nsid=%u, length=%zu, value_size=%zu; "
                "建议将盘框 nsid=%u 的 valueSize 与 value_size_kb_nsid%u 配置为 %zu%s\n",
                blockId, nsid, length, expectValueSize, nsid, nsid, suggest_kb, roundNote);
        } else {
            NDS_ERROR_PRINT(
                "length > value_size: blockId=%lu, nsid=%u, length=%zu, value_size=%zu; "
                "建议将盘框 nsid=%u 的 valueSize 与 value_size_kb 配置为 %zu，"
                "或新增 value_size_kb_nsid%u=%zu%s\n",
                blockId, nsid, length, expectValueSize, nsid, suggest_kb, nsid, suggest_kb,
                roundNote);
        }
        return {};
    } else if (length < expectValueSize) {
        // 短写补齐：pair 覆盖范围直接扩到 [blockAddr, blockAddr+value_size)，
        // 单调递增一次写满整个 slot（尾部为 buffer 残留数据，don't-care，
        // 不做任何清零动作）。策略演进见 appendRegionPairs 上方的注释。
        if (offset != 0) {
            NDS_ERROR_PRINT(
                "non-zero write offset not allowed for padded write: blockId=%lu, offset=%zu\n",
                blockId, offset);
            return {};
        }
        // 补齐后 RDMA READ 范围扩到 value_size。优先 contiguous 扩展；block 落在
        // 池末尾、扩范围越出所属 MR 时，pad 段改借该 MR 头部已注册内存（同一
        // mr_id，内容 don't-care），让末页也能写满 slot。详见批量路径同名注释。
        uint64_t blockAddrVal = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(blockAddr));
        if (find_mr_index_nolog(blockAddrVal, expectValueSize) >= 0) {
            NDS_DEBUG_PRINT(
                "padded write: blockId=%lu, nsid=%u, dataLen=%zu, writing full slot=%zu\n", blockId,
                nsid, length, expectValueSize);
            pairs.reserve((expectValueSize + lengthPerBatch - 1) / lengthPerBatch);
            appendRegionPairs(blockId, blockIdHigh, blockAddr, 0, expectValueSize, lengthPerBatch,
                              pairs);
        } else {
            const uint64_t mrBase = g_mr_ranges[mr].first;
            const uint64_t mrSize = g_mr_ranges[mr].second;
            if (mrSize < expectValueSize) {
                NDS_ERROR_PRINT(
                    "padded write: blockId=%lu, addr=%p, value_size=%zu exceeds "
                    "its MR [%llu,+%llu) and MR is smaller than value_size; "
                    "cannot borrow MR head for padding\n",
                    blockId, (void*)blockAddr, expectValueSize, (unsigned long long)mrBase,
                    (unsigned long long)mrSize);
                return {};
            }
            NDS_DEBUG_PRINT(
                "padded write borrow MR head: blockId=%lu, nsid=%u, dataLen=%zu, "
                "padLen=%zu borrowed from mr[%d] base\n",
                blockId, nsid, length, expectValueSize - length, mr);
            pairs.reserve((expectValueSize + lengthPerBatch - 1) / lengthPerBatch + 1);
            appendRegionPairs(blockId, blockIdHigh, blockAddr, 0, length, lengthPerBatch, pairs);
            uint8_t* padSrc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(mrBase));
            appendRegionPairs(blockId, blockIdHigh, padSrc, length, expectValueSize - length,
                              lengthPerBatch, pairs);
        }
    } else {
        debugNoteShortTail(blockId, nsid, length, lengthPerBatch);
        pairs.reserve((length + lengthPerBatch - 1) / lengthPerBatch);
        appendRegionPairs(blockId, blockIdHigh, blockAddr, offset, length, lengthPerBatch, pairs);
    }

    NDS_DEBUG_PRINT(
        "****** prepareNdsReqs, is_read: %d, lengthPerBatch: %zu, maxAddrNum: %zu, pairCount: "
        "%zu\n",
        read, lengthPerBatch, maxAddrNum, pairs.size());

    // 单个 NDS_REQ 的 addr_pair 数量不能超过 maxAddrNum，按 maxAddrNum 拆分为多个 NDS_REQ；
    // 但每个 NDS_REQ 仍只对应同一个逻辑 block。
    std::vector<NdsReqWrapper> reqs;
    reqs.reserve((pairs.size() + maxAddrNum - 1) / maxAddrNum);
    for (size_t pairIdx = 0; pairIdx < pairs.size(); pairIdx += maxAddrNum) {
        size_t chunkSize = std::min(pairs.size() - pairIdx, maxAddrNum);
        NdsReqWrapper wrapper;
        wrapper.req = NDS_REQ{};
        wrapper.req.addr_num = static_cast<uint16_t>(chunkSize);
        wrapper.req.task_id = next_task_id();
        wrapper.req.mr_id = mr;
        wrapper.req.nsid = nsid;
        for (size_t j = 0; j < chunkSize; ++j) {
            wrapper.req.addr_pair_list[j] = pairs[pairIdx + j];
        }
        wrapper.total_blocks = 1;
        reqs.push_back(wrapper);
    }
    return reqs;
}

// 滑动窗口模型：调用线程内先批量提交 win 个 REQ 填充管线，再逐个 wait 最老的 +
// submit 下一个，每线程同时有 g_max_inflight_reqs 个 REQ 在飞。
// 替代了旧两阶段和信号量方案。信号量方案因每线程 submit→wait 严格串行，
// 每线程永远 1 个在飞（无论信号量多大），8 线程最多 8 并发，RDMA QP 空闲。
// 滑动窗口先提交 win 个（不 wait），形成 win 个在飞管线；之后 loop：
// wait 最老的（完成一个）→ submit 下一个（填回空位），管线保持 win 个在飞。
// task_id 由全局原子计数器分配（见 next_task_id），每个 REQ 唯一。
// 返回失败 REQ 数（0 为全部成功）。
int32_t submitAsyncAndWait(std::vector<NdsReqWrapper>& reqs, bool read)
{
    int32_t error_num = 0;
    size_t win = g_max_inflight_reqs;
    if (win < 1) win = 1;
    if (win > reqs.size()) win = reqs.size();

    const size_t n = reqs.size();
    std::vector<int8_t> submitOk(n, 0);  // 0=not submitted, 1=ok, -1=failed
    size_t submitIdx = 0;

    // Phase 1: 填充管线 — 先提交 win 个 REQ（不 wait）
    for (; submitIdx < n && submitIdx < win; submitIdx++) {
        auto& w = reqs[submitIdx];
        NDS_DEBUG_PRINT("****** submitAsyncAndWait phase1 submit, reqs id: %lu \n", w.req.task_id);
        int32_t ret = read ? nds_kv_get(&w.req) : nds_kv_put(&w.req);
        if (ret != 0) {
            NDS_ERROR_PRINT(
                "nds kv operation failed, is read:%d, mrId:%u, blockId:%lu, blockAddr:%lu, "
                "offset:%u, length:%u\n",
                read, w.req.mr_id, w.req.addr_pair_list[0].block_id.key_low,
                w.req.addr_pair_list[0].hbm_addr, w.req.addr_pair_list[0].offset,
                w.req.addr_pair_list[0].length);
            error_num++;
            submitOk[submitIdx] = -1;
        } else {
            submitOk[submitIdx] = 1;
        }
    }

    // Phase 2: 滑动窗口 — wait 最老的，submit 下一个
    for (size_t i = 0; i < n; i++) {
        if (submitOk[i] == 1) {
            auto& w = reqs[i];
            NDS_DEBUG_PRINT("****** submitAsyncAndWait nds_kv_wait start, reqs id: %lu \n",
                            w.req.task_id);
            CODE_SET code_set = nds_kv_wait(w.req.task_id, w.total_blocks);
            NDS_DEBUG_PRINT(
                "****** submitAsyncAndWait nds_kv_wait end, reqs id: %lu, batch_size: %zu, "
                "key_size: %zu \n",
                w.req.task_id, code_set.batch_code_set.size(), code_set.key_code_set.size());
            if (checkCodeSet(code_set, w.req.task_id, read) != 0) {
                NDS_ERROR_PRINT(
                    "nds_kv_wait failed, is read:%d, blockId:%lu, blockAddr:%lu, offset:%u, "
                    "length:%u\n",
                    read, w.req.addr_pair_list[0].block_id.key_low,
                    w.req.addr_pair_list[0].hbm_addr, w.req.addr_pair_list[0].offset,
                    w.req.addr_pair_list[0].length);
                error_num++;
            }
        }
        // submitOk[i] == -1: 提交失败，已在 Phase 1/2 记过 error，跳过 wait
        // submitOk[i] == 0:  未提交（win > n 导致管线未满即 wait 完），不等待

        // 管线有空位就填下一个
        if (submitIdx < n) {
            auto& w = reqs[submitIdx];
            NDS_DEBUG_PRINT("****** submitAsyncAndWait phase2 submit, reqs id: %lu \n",
                            w.req.task_id);
            int32_t ret = read ? nds_kv_get(&w.req) : nds_kv_put(&w.req);
            if (ret != 0) {
                NDS_ERROR_PRINT(
                    "nds kv operation failed, is read:%d, mrId:%u, blockId:%lu, blockAddr:%lu, "
                    "offset:%u, length:%u\n",
                    read, w.req.mr_id, w.req.addr_pair_list[0].block_id.key_low,
                    w.req.addr_pair_list[0].hbm_addr, w.req.addr_pair_list[0].offset,
                    w.req.addr_pair_list[0].length);
                error_num++;
                submitOk[submitIdx] = -1;
            } else {
                submitOk[submitIdx] = 1;
            }
            submitIdx++;
        }
    }
    return error_num;
}

int32_t NDS::get(uint64_t blockId, uint16_t blockIdHigh, uint8_t* blockAddr, size_t offset,
                 size_t length, uint32_t nsid)
{
    auto reqs = prepareNdsReqs(blockId, blockIdHigh, blockAddr, offset, length, nsid, true);
    if (reqs.empty()) {
        NDS_ERROR_PRINT("prepareNdsReqs failed (empty reqs): blockId=%lu, offset=%zu, length=%zu\n",
                        blockId, offset, length);
        return -1;
    }
    int ret = 0;
    for (uint64_t i = 0; i < reqs.size(); ++i) {
        ret = nds_kv_get(&reqs[i].req);
        if (ret != 0) {
            NDS_ERROR_PRINT(
                "nds_kv_get failed, blockId:%lu, idx:%lu, blockAddr:%p, offset:%lu, length:%u, "
                "nsid: %d\n",
                blockId, i, blockAddr, offset, length, nsid);
            break;
        }
        CODE_SET code_set = nds_kv_wait(reqs[i].req.task_id, reqs[i].total_blocks);
        if (checkCodeSet(code_set, reqs[i].req.task_id, true) != 0) {
            NDS_ERROR_PRINT(
                "nds_kv_wait failed, blockId:%lu, idx:%lu, blockAddr:%p, offset:%lu, length:%lu\n",
                blockId, i, blockAddr, offset, length);
            return -1;
        }
    }
    return ret;
}

int32_t NDS::batchGet(std::vector<uint64_t> blockIds, std::vector<uint16_t> blockIdHighs,
                      std::vector<uint8_t*> blockAddrs, std::vector<size_t> offsets,
                      std::vector<size_t> lengths, std::vector<uint32_t> nsids)
{
    std::vector<NdsReqWrapper> allReqs =
        prepareNdsReqsBatch(blockIds, blockIdHighs, blockAddrs, offsets, lengths, nsids, true);
    if (allReqs.empty() && !blockIds.empty()) {
        NDS_ERROR_PRINT("prepareNdsReqsBatch failed (empty reqs): blockCount=%zu\n",
                        blockIds.size());
        return -1;
    }
    NDS_DEBUG_PRINT("****** NDS::batchGet start \n");
    int32_t ret = submitAsyncAndWait(allReqs, true);
    NDS_DEBUG_PRINT("****** NDS::batchGet end \n");
    return ret;
}

int32_t NDS::put(uint64_t blockId, uint16_t blockIdHigh, uint8_t* blockAddr, size_t offset,
                 size_t length, uint32_t nsid)
{
    auto reqs = prepareNdsReqs(blockId, blockIdHigh, blockAddr, offset, length, nsid, false);
    if (reqs.empty()) {
        NDS_ERROR_PRINT("prepareNdsReqs failed (empty reqs): blockId=%lu, offset=%zu, length=%zu\n",
                        blockId, offset, length);
        return -1;
    }
    int ret = 0;
    for (uint64_t i = 0; i < reqs.size(); ++i) {
        ret = nds_kv_put(&reqs[i].req);
        if (ret != 0) {
            NDS_ERROR_PRINT(
                "nds_kv_put failed, blockId:%lu, idx:%lu, blockAddr:%p, offset:%lu, length:%u, "
                "nsid: %d\n",
                blockId, i, blockAddr, offset, length, nsid);
            break;
        }
        CODE_SET code_set = nds_kv_wait(reqs[i].req.task_id, reqs[i].total_blocks);
        if (checkCodeSet(code_set, reqs[i].req.task_id, false) != 0) {
            NDS_ERROR_PRINT(
                "nds_kv_wait failed, blockId:%lu, idx:%lu, blockAddr:%p, offset:%lu, length:%lu\n",
                blockId, i, blockAddr, offset, length);
            return -1;
        }
    }
    return ret;
}

int32_t NDS::batchPut(std::vector<uint64_t> blockIds, std::vector<uint16_t> blockIdHighs,
                      std::vector<uint8_t*> blockAddrs, std::vector<size_t> offsets,
                      std::vector<size_t> lengths, std::vector<uint32_t> nsids)
{
    std::vector<NdsReqWrapper> allReqs =
        prepareNdsReqsBatch(blockIds, blockIdHighs, blockAddrs, offsets, lengths, nsids, false);
    if (allReqs.empty() && !blockIds.empty()) {
        NDS_ERROR_PRINT("prepareNdsReqsBatch failed (empty reqs): blockCount=%zu\n",
                        blockIds.size());
        return -1;
    }
    int32_t ret = submitAsyncAndWait(allReqs, false);
    return ret;
}
