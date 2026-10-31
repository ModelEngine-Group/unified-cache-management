/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: KV OSD Client Agent实现
*/

// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#include "KVClientAgentImpl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "include/ceph_assert.h"
#include "common/async/context_pool.h"
#include "common/ceph_argparse.h"
#include "common/ceph_context.h"
#include "common/common_init.h"
#include "common/config.h"
#include "common/errno.h"
#include "include/ceph_fs.h"
#include "include/encoding.h"
#include "messages/MOSDMap.h"
#include "mon/MonClient.h"
#include "msg/Dispatcher.h"
#include "msg/Messenger.h"
#include "osd/OSDMap.h"

namespace agent {
namespace {

// OSD Agent 固定监听端口, 与 KVOSDAgent 中 KV_OSD_AGENT_DEFAULT_PORT 一致.
// OSDMap 中地址端口为 0 时用此值补齐.
constexpr int KV_OSD_AGENT_DEFAULT_PORT = 6800;

struct ViewRefresherState {
    std::thread refresh_thread;
    std::condition_variable cv;
    std::mutex refresher_lock;      // 保护 cv 等待/唤醒
    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};

    // Round-Robin 游标
    std::atomic<int> next_dpu_idx{0};

    // 配置参数
    std::chrono::seconds refresh_interval{KV_VIEW_REFRESH_INTERVAL_DEFAULT};
    std::chrono::seconds fetch_timeout{KV_VIEW_FETCH_TIMEOUT_DEFAULT};

    // 异步拉取状态
    // pending_fetch_osd >= 0 表示正在等待该 OSD 的MOSDMap 响应
    std::atomic<int> pending_fetch_osd{-1};

    // 连接缓存: osd_id -> ConnectionRef
    std::map<int, ConnectionRef> conn_cache;
    std::mutex conn_cache_lock;     // 独立锁, 避免与 KVOSDClientAgent::lock 嵌套
};

class KVOSDClientAgent final : public KVClientAgentImpl, public Dispatcher {
public:
    KVOSDClientAgent()
        : Dispatcher(nullptr)
    {}

    ~KVOSDClientAgent() override
    {
        shutdown();
    }

    // Client Agent 初始化含 MonClient 连接, OSD 连接缓存等完整流程
    // NOLINTBEGIN(huge_method)
    int init(const KVClientAgentConfig& config,
             AgentLogger *agent_logger) override
    {
        std::lock_guard guard(lock);
        if (initialized) {
            return -EALREADY;
        }
        logger = agent_logger;

        CephInitParameters params(CEPH_ENTITY_TYPE_CLIENT);
        if (!params.name.from_str(config.client_name)) {
            return -EINVAL;
        }
        context = boost::intrusive_ptr<CephContext>(
            common_preinit(params, CODE_ENVIRONMENT_LIBRARY, 0), false);
        int r = context->_conf.set_val("mon_host", config.mon_host);
        if (r < 0) {
            return r;
        }
        r = context->_conf.set_val("key", config.client_key);
        if (r < 0) {
            return r;
        }
        context->_conf.apply_changes(nullptr);
        common_init_finish(context.get());
        Dispatcher::cct = context.get();

        pool = std::make_unique<ceph::async::io_context_pool>(1);
        pool->start(1);
        monc = std::make_unique<MonClient>(context.get(), *pool);
        r = monc->build_initial_monmap();
        if (r < 0) {
            return r;
        }
        messenger.reset(
            Messenger::create_client_messenger(context.get(), "kv-client-agent"));
        if (!messenger) {
            return -ENOMEM;
        }
        monc->set_messenger(messenger.get());
        messenger->set_auth_client(monc.get());
        messenger->add_dispatcher_tail(this);
        messenger->start();
        messenger_started = true;
        monc->set_want_keys(CEPH_ENTITY_TYPE_MON | CEPH_ENTITY_TYPE_OSD);
        r = monc->init();
        if (r < 0) {
            return r;
        }
        monc_started = true;
        r = monc->authenticate(
            context->_conf.get_val<std::chrono::seconds>(
                "client_mount_timeout").count());
        if (r < 0) {
            return r;
        }
        // 等待 OSD rotating key 就绪, 否则首次连接 OSD Agent 时
        // CephX 认证会因缺少 rotating key 被拒绝, 导致连接 reset.
        r = monc->wait_auth_rotating(
            context->_conf.get_val<std::chrono::seconds>(
                "client_mount_timeout").count());
        if (r < 0) {
            return r;
        }

        // 保存配置参数供 view refresher 使用
        view_refresh_interval_cfg = 
            std::chrono::seconds(config.view_refresh_interval);
        view_fetch_timeout_cfg = 
            std::chrono::seconds(config.view_fetch_timeout);

        initialized = true;
        return 0;
    }
    // NOLINTEND(huge_method)

    int start() override
    {
        std::lock_guard guard(lock);
        if (!initialized) {
            return -ENOTCONN;
        }
        if (started) {
            return -EALREADY;
        }
        // ① 一次性订阅 OSDMap (MON 推送首次 OSDMap 后不再持续推送)
        monc->sub_want("osdmap", 0, CEPH_SUBSCRIBE_ONETIME);
        monc->renew_subs();
        started = true;
        AGENT_LOG(logger, LogLevel::INFO, "requested one OSD map");

        // ② 启动视图刷新线程
        start_view_refresher();
        return 0;
    }

    uint64_t get_epoch() const override
    {
        std::lock_guard guard(lock);
        return osdmap.get_epoch();
    }

    std::vector<KVClientNodeView> get_view() const override
    {
        std::lock_guard guard(lock);
        std::vector<KVClientNodeView> result;
        for (int osd = 0; osd < osdmap.get_max_osd(); ++osd) {
            if (!osdmap.exists(osd)) {
                continue;
            }
            KVClientNodeView node{
                .osd_id = osd,
                .up = osdmap.is_up(osd),
                .in = osdmap.is_in(osd),
            };
            for (const auto& address : osdmap.get_addrs(osd).v) {
                const auto ip = address.ip_only_to_str();
                if (std::find(node.addrs.begin(), node.addrs.end(), ip) ==
                    node.addrs.end()) {
                    node.addrs.push_back(ip);
                }
            }
            result.push_back(std::move(node));
        }
        return result;
    }

    std::vector<KVClientNodeView> get_view_by_ns(
        const std::string& ns, bool up, bool in) const override
    {
        std::lock_guard guard(lock);
        std::vector<KVClientNodeView> result;

        int64_t pool_id = osdmap.lookup_pg_pool_name(ns);
        if (pool_id < 0) {
            AGENT_LOG(logger, LogLevel::WARNING, "ns view: pool name not found: " + ns);
            return result;
        }

        int rule_id = osdmap.get_pool_crush_rule(pool_id);

        std::set<int> roots;
        osdmap.crush->find_takes_by_rule(rule_id, &roots);
        if (roots.empty()) {
            AGENT_LOG(logger, LogLevel::WARNING, "ns view: no crush roots for rule " + std::to_string(rule_id));
            return result;
        }

        std::set<int32_t> pool_osds;
        for (int root : roots) {
            std::set<int> children;
            osdmap.crush->get_all_children(root, &children);
            pool_osds.insert(children.begin(), children.end());
        }

        for (int32_t osd : pool_osds) {
            if (osd < 0 || !osdmap.exists(osd)) {
                continue;
            }
            if (osdmap.is_up(osd) != up || osdmap.is_in(osd) != in) {
                continue;
            }
            KVClientNodeView node{
                .osd_id = osd,
                .up = osdmap.is_up(osd),
                .in = osdmap.is_in(osd),
            };
            for (const auto& address : osdmap.get_addrs(osd).v) {
                const auto ip = address.ip_only_to_str();
                if (std::find(node.addrs.begin(), node.addrs.end(), ip) ==
                    node.addrs.end()) {
                    node.addrs.push_back(ip);
                }
            }
            result.push_back(std::move(node));
        }
        return result;
    }

    std::vector<KVClientNodeView> get_up_in_by_ns(
        const std::string& ns) const override
    {
        return get_view_by_ns(ns, true, true);
    }

    std::vector<KVClientNodeView> get_down_in_by_ns(
        const std::string& ns) const override
    {
        return get_view_by_ns(ns, false, true);
    }

    std::vector<KVClientNodeView> get_up_out_by_ns(
        const std::string& ns) const override
    {
        return get_view_by_ns(ns, true, false);
    }

    std::vector<KVClientNodeView> get_down_out_by_ns(
        const std::string& ns) const override
    {
        return get_view_by_ns(ns, false, false);
    }

    std::vector<KVClientNodeView> get_in() const override
    {
        std::lock_guard guard(lock);
        std::vector<KVClientNodeView> result;
        for (int osd = 0; osd < osdmap.get_max_osd(); ++osd) {
            if (!osdmap.exists(osd)) {
                continue;
            }
            if (!osdmap.is_in(osd)) {
                continue;
            }
            KVClientNodeView node{
                .osd_id = osd,
                .up = osdmap.is_up(osd),
                .in = osdmap.is_in(osd),
            };
            for (const auto& address : osdmap.get_addrs(osd).v) {
                const auto ip = address.ip_only_to_str();
                if (std::find(node.addrs.begin(), node.addrs.end(), ip) ==
                    node.addrs.end()) {
                    node.addrs.push_back(ip);
                }
            }
            result.push_back(std::move(node));
        }
        return result;
    }

    std::vector<KVClientNodeView> get_in_by_ns(
        const std::string& ns) const override
    {
        std::lock_guard guard(lock);
        std::vector<KVClientNodeView> result;
        
        int64_t pool_id = osdmap.lookup_pg_pool_name(ns);
        if (pool_id < 0) {
            AGENT_LOG(logger, LogLevel::WARNING,
                        "get_in_by_ns: pool name not found: " + ns);
            return result;
        }

        int rule_id = osdmap.get_pool_crush_rule(pool_id);
        std::set<int> roots;
        osdmap.crush->find_takes_by_rule(rule_id, &roots);
        if (roots.empty()) {
            AGENT_LOG(logger, LogLevel::WARNING,
                        "get_in_by_ns: no crush roots for rule " +
                        std::to_string(rule_id));
            return result;
        }

        std::set<int32_t> pool_osds;
        for (int root : roots) {
            std::set<int> children;
            osdmap.crush->get_all_children(root, &children);
            pool_osds.insert(children.begin(), children.end());
        }

        for (int32_t osd : pool_osds) {
            if (osd < 0 || !osdmap.exists(osd)) {
                continue;
            }
            if (!osdmap.is_in(osd)) {
                continue;
            }
            KVClientNodeView node{
                .osd_id = osd,
                .up = osdmap.is_up(osd),
                .in = osdmap.is_in(osd),
            };
            for (const auto& address : osdmap.get_addrs(osd).v) {
                const auto ip = address.ip_only_to_str();
                if (std::find(node.addrs.begin(), node.addrs.end(), ip) ==
                    node.addrs.end()) {
                    node.addrs.push_back(ip);
                }
            }
            result.push_back(std::move(node));
        }
        return result;
    }

    std::vector<std::string> get_all_ns_names() const override
    {
        std::lock_guard guard(lock);
        std::vector<std::string> result;
        const auto& pools = osdmap.get_pools();
        for (const auto& [pool_id, pool] : pools) {
            auto name = osdmap.get_pool_name(pool_id);
            if (!name.empty()) {
                result.push_back(std::move(name));
            }
        }
        return result;
    }

    // --- 视图刷新 ---

    int start_view_refresh() override
    {
        // start() 中已自动启动, 此方法供独立调用场景使用
        if (view_refresher && view_refresher->running.load()) {
            return -EALREADY;
        }
        // 正常流程中 start() 已调用 start_view_refresher, 此处不应再调
        return 0;
    }

    int stop_view_refresh() override
    {
        stop_view_refresher();
        return 0;
    }

    // --- ms_dispatch ---

    bool ms_dispatch(Message *message) override
    {
        AGENT_LOG(logger, LogLevel::INFO,
                    "ms_dispatch: type=" + std::to_string(message->get_type()) +
                    " name=" + std::string(message->get_type_name()) +
                    " src=" + std::string(message->get_source().type_str()) + "." + std::to_string(message->get_source().num()) +
                    " pending_fetch_osd=" + 
                        std::to_string(view_refresher ? 
                            view_refresher->pending_fetch_osd.load() : -2));
        if (message->get_type() == CEPH_MSG_OSD_MAP) {
            return handle_osdmap_message(static_cast<MOSDMap *>(message));
        }
        AGENT_LOG(logger, LogLevel::WARNING,
                    "ms_dispatch: unexpected message type=" +
                    std::to_string(message->get_type()) +
                    " name=" + std::string(message->get_type_name()));
        return false;
    }

    bool ms_handle_reset(Connection *c) override
    {
        notify_connection_failure(c);
        return false;
    }
    void ms_handle_remote_reset(Connection *) override {}
    bool ms_handle_refused(Connection *c) override
    {
        notify_connection_failure(c);
        return false;
    }

private:
    // --- Connection failure notification ---

    // 连接被 reset/refused 时清除 pending fetch 并唤醒 refresher,
    // 否则 fetch_view_from_dpu 会一直等到超时才尝试下一个 DPU.
    void notify_connection_failure(Connection *c)
    {
        if (!view_refresher) return;
        int osd_id = view_refresher->pending_fetch_osd.load();
        if (osd_id < 0) return;
        AGENT_LOG(logger, LogLevel::WARNING,
                    "connection to OSD " + std::to_string(osd_id) +
                    " reset/refused (peer=" +
                    c->get_peer_addr().ip_only_to_str() + ")");
        view_refresher->pending_fetch_osd.store(-1);
        {
            std::lock_guard cl(view_refresher->conn_cache_lock);
            view_refresher->conn_cache.erase(osd_id);
        }
        std::lock_guard rl(view_refresher->refresher_lock);
        view_refresher->cv.notify_all();
    }

    // --- Phase 0: 提取 apply_osdmap_message() ---

    // 应用 OSDMap 消息到本地 osdmap (持 lock)
    // 返回 >0 表示应用后的新 epoch, 返回 0 表示未应用 (quiesced)
    uint64_t apply_osdmap_message(MOSDMap *osdmap_message)
    {
        std::lock_guard guard(lock);
        if (quiesced) {
            return 0;
        }

        for (epoch_t epoch = osdmap.get_epoch() + 1;
             epoch <= osdmap_message->get_last(); ++epoch) {
            auto full = osdmap_message->maps.find(epoch);
            if (full != osdmap_message->maps.end()) {
                osdmap.decode(full->second);
                continue;
            }
            auto incremental = osdmap_message->incremental_maps.find(epoch);
            if (incremental != osdmap_message->incremental_maps.end() &&
                osdmap.get_epoch() == epoch - 1) {
                OSDMap::Incremental update(incremental->second);
                osdmap.apply_incremental(update);
            }
        }

        AGENT_LOG(logger, LogLevel::INFO,
                    "applied OSD map epoch " +
                    std::to_string(osdmap.get_epoch()) +
                    " up " + std::to_string(osdmap.get_num_up_osds()) +
                    "/" + std::to_string(osdmap.get_num_osds()));
        return osdmap.get_epoch();
    }

    // 处理收到的 MOSDMap (MON 推送或 DPU 拉取响应)
    bool handle_osdmap_message(MOSDMap *message)
    {
        // ① 判断来源: OSD (DPU 拉取响应) vs MON (一次性推送)
        const auto source_name = message->get_source();
        bool from_dpu = source_name.is_osd();
        int source_osd = from_dpu ? source_name.num() : -1;

        AGENT_LOG(logger, LogLevel::INFO,
                    "handle_osdmap_message: src=" + std::string(source_name.type_str()) + "." + std::to_string(source_name.num()) +
                    " from_dpu=" + std::to_string(from_dpu) +
                    " source_osd=" + std::to_string(source_osd) +
                    " maps=" + std::to_string(message->maps.size()) +
                    " incremental_maps=" +
                        std::to_string(message->incremental_maps.size()) +
                    " newest_map=" + std::to_string(message->newest_map) +
                    " data_len=" + std::to_string(message->get_data_len()) +
                    " pending_fetch_osd=" + 
                        std::to_string(view_refresher ? 
                            view_refresher->pending_fetch_osd.load() : -2));

        // ② 记录旧 epoch (持 lock)
        uint64_t old_epoch;
        {
            std::lock_guard guard(lock);
            old_epoch = osdmap.get_epoch();
        }

        // ③ 应用 OSDMap (apply_osdmap_message 内部持 lock)
        //    注意: 若 apply_osdmap_message 内部 decode/apply_incremental 异常
        //    导致未返回, 则 ④ 的 CAS 不会执行, fetch_view_from_dpu 会超时.
        //    因此 CAS 清除 pending 的逻辑不依赖 apply 结果, 放在 put() 之后.
        uint64_t new_epoch = apply_osdmap_message(message);
        message->put();

        AGENT_LOG(logger, LogLevel::INFO,
                    "handle_osdmap_message: applied, old_epoch=" +
                    std::to_string(old_epoch) +
                    " new_epoch=" + std::to_string(new_epoch) +
                    " from_dpu=" + std::to_string(from_dpu));
        
        // ④ 如果是 DPU 拉取响应, 清除 pending 并唤醒 refresh thread
        //    (无论 epoch 是否推进, "up to date" 响应也算成功)
        if (from_dpu && view_refresher) {
            int expected = source_osd;
            if (view_refresher->pending_fetch_osd.compare_exchange_strong(expected, -1)) {
                AGENT_LOG(logger, LogLevel::INFO,
                            "handle_osdmap_message: CAS succeeded, "
                            "clearing pending_fetch_osd and notifying");
                std::lock_guard rl(view_refresher->refresher_lock);
                view_refresher->cv.notify_all();
            } else {
                AGENT_LOG(logger, LogLevel::WARNING,
                            "handle_osdmap_message: CAS FAILED"
                            " (expected pending=" + std::to_string(source_osd) +
                            ", actual=" +
                                std::to_string(expected) +
                            "), response from DPU will be ignored by waiter");
            }
        }

        if (new_epoch == 0 || new_epoch <= old_epoch) {
            return true;
        }

        // ⑤ 不主动通知上层 (上层通过 get_view() 周期检查)
        return true;
    }

    // --- Phase 2: ViewRefresher 线程生命周期 ---

    void start_view_refresher()
    {
        view_refresher = std::make_unique<ViewRefresherState>();
        view_refresher->refresh_interval = view_refresh_interval_cfg;
        view_refresher->fetch_timeout = view_fetch_timeout_cfg;
        view_refresher->running = true;
        view_refresher->stop_requested = false;
        view_refresher->next_dpu_idx = 0;
        view_refresher->pending_fetch_osd = -1;
        view_refresher->refresh_thread =
            std::thread(&KVOSDClientAgent::view_refresh_loop, this);
        AGENT_LOG(logger, LogLevel::INFO,
                    "view refresh thread started (interval=" +
                    std::to_string(view_refresher->refresh_interval.count()) +
                    "s, timeout=" +
                    std::to_string(view_refresher->fetch_timeout.count()) + "s)");
    }

    void stop_view_refresher()
    {
        if (!view_refresher) return;

        view_refresher->stop_requested = true;
        {
            std::lock_guard rl(view_refresher->refresher_lock);
            view_refresher->cv.notify_all();
        }
        if (view_refresher->refresh_thread.joinable()) {
            view_refresher->refresh_thread.join();
        }
        view_refresher->running = false;

        // 清理连接缓存
        {
            std::lock_guard cl(view_refresher->conn_cache_lock);
            view_refresher->conn_cache.clear();
        }

        AGENT_LOG(logger, LogLevel::INFO, "view refresh thread stopped");
        view_refresher.reset();
    }

    // --- Phase 2: 线程主循环 ---

    void view_refresh_loop()
    {
        AGENT_LOG(logger, LogLevel::INFO, "view refresh loop running");

        while (!view_refresher->stop_requested.load()) {
            // ① 等待周期或被 stop 提前唤醒
            {
                std::unique_lock<std::mutex> lk(view_refresher->refresher_lock);
                view_refresher->cv.wait_for(
                    lk, view_refresher->refresh_interval,
                    [this] { return view_refresher->stop_requested.load(); });
                AGENT_LOG(logger, LogLevel::DEBUG,
                            "refresh loop woke, pending=" +
                            std::to_string(
                                view_refresher->pending_fetch_osd.load()));
            }
            if (view_refresher->stop_requested.load()) break;

            // ② 获取 UP IN DPU 列表
            std::vector<int> dpu_list;
            extract_up_in_dpu_list(dpu_list);
            if (dpu_list.empty()) {
                AGENT_LOG(logger, LogLevel::INFO,
                            "OSD map not yet received or no UP IN DPU, skipping");
                continue;
            }

            // ③ Round-Robin 选择 DPU 并拉取
            bool success = false;
            int start_idx = view_refresher->next_dpu_idx.load();
            int count = static_cast<int>(dpu_list.size());

            for (int i = 0; i < count; ++i) { 
                int idx = (start_idx + i) % count;
                int target_osd = dpu_list[idx];

                success = fetch_view_from_dpu(target_osd);
                if (success) {
                    // ④ 游标更新: 成功 DPU 索引 +1 取模
                    view_refresher->next_dpu_idx.store((idx + 1) % count);
                    break;
                }
                // 失败继续尝试下一个 DPU
            }

            if (!success) {
                // 全部 DPU 失败, 游标保持不变, 下一轮从同一位置重试
                AGENT_LOG(logger, LogLevel::WARNING,
                            "all DPU view fetch failed, will retry next cycle");
            }
        }

        AGENT_LOG(logger, LogLevel::INFO, "view refresh loop exiting");
    }

    // --- Phase 3: extract_up_in_dpu_list ---

    void extract_up_in_dpu_list(std::vector<int>& dpu_list)
    {
        dpu_list.clear();

        // 持 lock 保护 osdmap
        std::lock_guard guard(lock);

        epoch_t epoch = osdmap.get_epoch();
        if (epoch == 0) {
            // 首次 OSDMap 尚未到达, 等 MON 推送
            return;
        }

        int max_osd = osdmap.get_max_osd();
        for (int osd_id = 0; osd_id < max_osd; ++osd_id) {
            if (!osdmap.exists(osd_id)) continue;
            if (!osdmap.is_up(osd_id)) continue;
            if (!osdmap.is_in(osd_id)) continue;

            // 验证地址可达性
            auto addrs = osdmap.get_addrs(osd_id);
            if (addrs.v.empty()) continue;

            // 只打印第一个 DPU IP, 这是实际用于连接的地址
            AGENT_LOG(logger, LogLevel::INFO,
                        "UP IN OSD " + std::to_string(osd_id) +
                        " first_dpu_ip=" + addrs.v[0].get_legacy_str() +
                        " (total addrs: " +
                            std::to_string(addrs.v.size()) + ")");
            
            dpu_list.push_back(osd_id);
        }

        AGENT_LOG(logger, LogLevel::DEBUG,
                    "extracted " + std::to_string(dpu_list.size()) +
                    " UP IN DPUs, epoch=" + std::to_string(epoch));
    }

    // --- Phase 4: 连接管理 ---

    ConnectionRef get_or_create_connection(int osd_id)
    {
        // ① 先查缓存
        {
            std::lock_guard cl(view_refresher->conn_cache_lock);
            auto it = view_refresher->conn_cache.find(osd_id);
            if (it != view_refresher->conn_cache.end() && it->second) {
                return it->second;
            }
        }

        // ② 缓存未命中, 从 osdmap 获取地址并建连
        entity_addrvec_t addrs;
        {
            std::lock_guard guard(lock);
            if (!osdmap.exists(osd_id)) {
                return nullptr;
            }
            addrs = osdmap.get_addrs(osd_id);
        }
        if (addrs.v.empty()) {
            AGENT_LOG(logger, LogLevel::WARNING,
                        "OSD " + std::to_string(osd_id) +
                        " has empty addrs in osdmap");
            return nullptr;
        }

        // 仅使用第一个 DPU IP 连接, 与 OSD Agent bind 的地址保持一致.
        // 若失败则由 view_refresh_loop 选择下一个 DPU重试.
        entity_addrvec_t first_dpu_addr;
        first_dpu_addr.v.push_back(addrs.v[0]);
        if (first_dpu_addr.v[0].get_port() == 0) {
            first_dpu_addr.v[0].set_port(KV_OSD_AGENT_DEFAULT_PORT);
        }

        AGENT_LOG(logger, LogLevel::INFO,
                    "connecting to OSD " + std::to_string(osd_id) +
                    " first_dpu_ip=" +
                        first_dpu_addr.v[0].get_legacy_str() +
                    " (total addrs in osdmap: " +
                        std::to_string(addrs.v.size()) + ")");

        addrs = first_dpu_addr;

        // ③ 创建连接
        auto conn = messenger->connect_to_osd(addrs);
        if (!conn) {
            AGENT_LOG(logger, LogLevel::WARNING,
                        "failed to create connection to OSD " +
                            std::to_string(osd_id));
            return nullptr;
        }

        // 给异步连接握手一点时间, 否则在连接就绪前发送的消息
        // 可能在 CephX 认证完成前被处理, 导致连接被 reset.
        // smoke test (kv-osdmap-pull-smoke.cc) 也采用了同样的策略.
        if (!conn->is_connected()) {
            AGENT_LOG(logger, LogLevel::DEBUG,
                        "waiting for async connection to OSD " +
                            std::to_string(osd_id) + "...");
            for (int i = 0; i < 50; ++i) {
                if (conn->is_connected()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            AGENT_LOG(logger, LogLevel::INFO,
                        "OSD " + std::to_string(osd_id) +
                        " connection ready: is_connected=" +
                        std::to_string(conn->is_connected()));
        }

        // ④ 存入缓存
        {
            std::lock_guard cl(view_refresher->conn_cache_lock);
            view_refresher->conn_cache[osd_id] = conn;
        }
        return conn;
    }

    // --- Phase 4: 单次拉取 (异步模式) ---

    bool fetch_view_from_dpu(int osd_id)
    {
        // ① 获取/创建连接
        auto conn = get_or_create_connection(osd_id);
        if (!conn) {
            AGENT_LOG(logger, LogLevel::WARNING,
                        "failed to connect to OSD " + std::to_string(osd_id) +
                        ", trying next");
            return false;
        }

        AGENT_LOG(logger, LogLevel::DEBUG,
                    "fetch_view_from_dpu: OSD " + std::to_string(osd_id) +
                    " conn=" + std::to_string(
                        reinterpret_cast<uintptr_t>(conn.get())) +
                    " peer=" + conn->get_peer_addr().ip_only_to_str() +
                    " is_connected=" +
                        std::to_string(conn->is_connected()));

        // ② 获取当前 epoch
        epoch_t current_epoch;
        {
            std::lock_guard guard(lock);
            current_epoch = osdmap.get_epoch();
        }

        // ③ 设置 pending_fetch_osd 标记 (必须在发送请求之前,否则若响应
        //    在设置前到达, handle_osdmap_message 的 CAS 会因 pending 仍为
        //    -1 而失败, 导致 notify 丢失, wait_for 死等到超时)
        view_refresher->pending_fetch_osd.store(osd_id);

        // ④ 发送 OSDMap pull 请求: 复用 MOSDMap, 将 current_epoch 编码进
        //    data bufferlist, maps 和 incremental_maps 保持为空.
        //    OSD Agent 通过 maps.empty() && incremental_maps.empty()
        //    && data_len > 0 判定为 pull 请求.
        auto req = ceph::make_message<MOSDMap>();
        ceph::buffer::list req_data;
        ceph::encode(current_epoch, req_data);
        req->set_data(req_data);
        int send_r = conn->send_message2(std::move(req));

        AGENT_LOG(logger, LogLevel::INFO,
                    "sent OSDMap pull request to OSD " + std::to_string(osd_id) +
                    ", current_epoch=" + std::to_string(current_epoch) +
                    " data_len=" + std::to_string(req_data.length()) +
                    " send_r=" + std::to_string(send_r) +
                    " is_connected=" +
                        std::to_string(conn->is_connected()));
                        
        // ⑤ 等待响应或超时 (异步等待: cv.wait_for)
        bool timed_out = false;
        {
            std::unique_lock<std::mutex> lk(view_refresher->refresher_lock);
            view_refresher->cv.wait_for(
                lk, view_refresher->fetch_timeout,
                [this, osd_id] {
                    return view_refresher->stop_requested.load() ||
                           view_refresher->pending_fetch_osd.load() != osd_id;
                });
            int pending_after = view_refresher->pending_fetch_osd.load();
            bool stop = view_refresher->stop_requested.load();
            AGENT_LOG(logger, LogLevel::INFO,
                        "wait_for woke: pending=" +
                            std::to_string(pending_after) +
                        " stop=" + std::to_string(stop) +
                        " osd_id=" + std::to_string(osd_id) +
                        " is_connected=" +
                            std::to_string(conn->is_connected()));
            // 检查是否超时 (pending 仍为 osd_id 说明未收到响应)
            if (pending_after == osd_id) {
                timed_out = true;
            }
        }

        if (timed_out) {
            // ⑥ 超时: 清除 pending, 清除可能已坏的连接, 尝试下一个 DPU
            view_refresher->pending_fetch_osd.store(-1);
            {
                std::lock_guard cl(view_refresher->conn_cache_lock);
                view_refresher->conn_cache.erase(osd_id);
            }
            AGENT_LOG(logger, LogLevel::WARNING,
                        "DPU " + std::to_string(osd_id) +
                        " view fetch timeout (timeout=" +
                            std::to_string(
                                view_refresher->fetch_timeout.count()) +
                        "s), is_connected=" +
                            std::to_string(conn->is_connected()) +
                        ", trying next");
            return false;
        }

        if (view_refresher->stop_requested.load()) {
            // 被 stop 唤醒
            view_refresher->pending_fetch_osd.store(-1);
            return false;
        }

        // ⑦ 收到响应 (ms_dispatch 已在 handle_osdmap_message 中 apply +
        //    清除 pending + cv.notify_all)
        AGENT_LOG(logger, LogLevel::DEBUG,
                    "view fetch from OSD " + std::to_string(osd_id) + " succeeded");
        return true;
    }

    // --- shutdown ---

    void shutdown()
    {
        // ① 先停视图刷新线程 (在 messenger 关闭前, 确保线程不再发消息)
        stop_view_refresher();

        {
            std::lock_guard guard(lock);
            quiesced = true;
        }
        if (monc_started) {
            monc->shutdown();
            monc_started = false;
        }
        if (messenger_started) {
            messenger->shutdown();
            messenger->wait();
            messenger_started = false;
        }
        if (pool) {
            pool->stop();
        }
        messenger.reset();
        monc.reset();
        pool.reset();
        context.reset();
        {
            std::lock_guard guard(lock);
            initialized = false;
            started = false;
        }
    }

    // --- 成员变量 ---

    mutable std::mutex lock;  // 保护 osdmap 及状态标志
    boost::intrusive_ptr<CephContext> context;
    std::unique_ptr<ceph::async::io_context_pool> pool;
    std::unique_ptr<MonClient> monc;
    std::unique_ptr<Messenger> messenger;
    OSDMap osdmap;
    AgentLogger *logger = nullptr;
    bool initialized = false;
    bool started = false;
    bool messenger_started = false;
    bool monc_started = false;
    bool quiesced = false;

    // 视图刷新
    std::unique_ptr<ViewRefresherState> view_refresher;
    std::chrono::seconds view_refresh_interval_cfg{KV_VIEW_REFRESH_INTERVAL_DEFAULT};
    std::chrono::seconds view_fetch_timeout_cfg{KV_VIEW_FETCH_TIMEOUT_DEFAULT};
};

} // namespace

std::unique_ptr<KVClientAgentImpl> create_kv_client_agent()
{
    return std::make_unique<KVOSDClientAgent>();
}

} // namespace agent