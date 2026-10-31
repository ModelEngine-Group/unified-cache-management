// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#include "KVAgentImpl.h"
#include "AgentLogger.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#include <random>

#include <boost/asio/steady_timer.hpp>
#include <boost/system/error_code.hpp>

#include "include/ceph_assert.h"
#include "common/async/context_pool.h"
#include "common/ceph_argparse.h"
#include "common/ceph_mutex.h"
#include "common/common_init.h"
#include "common/config.h"
#include "common/errno.h"
#include "include/encoding.h"
#include "messages/MOSDBeacon.h"
#include "messages/MOSDBoot.h"
#include "messages/MOSDMap.h"
#include "messages/MOSDPGCreate2.h"
#include "messages/MOSDPGCreated.h"
#include "mon/MonClient.h"
#include "msg/Messenger.h"
#include "osd/OSDMap.h"
#include "osd/osd_types.h"
#include "common/ceph_context.h"


namespace agent {

namespace {

// Boot 指数退避参数（编译期宏常量, 不通过C ABI配置传入）
constexpr int BOOT_RETRY_DELAY = 5;         // 秒, Boot 重试初始等待时间
constexpr int BOOT_RETRY_MAX_DELAY = 60;    // 秒, Boot 重试等待上限
// OSD Agent 固定监听端口, 配置未指定端口时使用此值
// constexpr int KV_OSD_AGENT_DEFAULT_PORT = 6800;
constexpr int KV_OSD_AGENT_DEFAULT_PORT = 65536;

int parse_public_addrs(const std::string& value, entity_addrvec_t *result,
                       AgentLogger *logger = nullptr)
{
    if (value.empty() || !result) {
        return -EINVAL;
    }

    entity_addrvec_t parsed;
    size_t begin = 0;
    while (begin < value.size()) {
        const auto comma = value.find(',', begin);
        const auto token = value.substr(begin, comma - begin);
        if (token.empty()) {
            return -EINVAL;
        }

        entity_addr_t address;
        if (!address.parse(token) || !address.is_ip()) {
            if (logger) {
                AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                              "parse_public_addrs: failed to parse '" + token + "'");
            }
            return -EINVAL;
        }
        address.set_type(entity_addr_t::TYPE_MSGR2);
        // 配置未指定端口时固定使用 6800, 确保 Client Agent 能从 OSDMap
        // 中获取到可连接的地址.
        if (address.get_port() == 0) {
            address.set_port(KV_OSD_AGENT_DEFAULT_PORT);
        }
        if (logger) {
            AGENT_LOG(logger, ::agent::LogLevel::INFO,
                        "parse_public_addrs: token='" + token +
                            "' parsed=" + address.get_legacy_str() +
                            " type=" + std::to_string(address.get_type()) +
                            " port=" + std::to_string(address.get_port()) +
                            " nonce=" + std::to_string(address.get_nonce()));
        }
        if (std::find(parsed.v.begin(), parsed.v.end(), address) !=
            parsed.v.end()) {
            return -EINVAL;
        }
        parsed.v.push_back(address);

        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
        if (begin == value.size()) {
            return -EINVAL;
        }
    }

    result->v = std::move(parsed.v);
    return 0;
}

enum class AgentState {
    DOWN,       // 初始状态 / 已知自身被 MON 标 down
    BOOT_SENT,  // 已发 MOSDBoot, 等待 OSDMap确认
    ACTIVE,     // MON 已确认 up, 周期发 MOSDBeacon
};

class KVOSDMapDispatcher final : public Dispatcher {
public:
    KVOSDMapDispatcher(CephContext *cct, MonClient *monc, AgentLogger *logger,
                       int osd_id,
                       std::function<void(epoch_t, bool)> notify_osdmap)
        : Dispatcher(cct), monc(monc), logger(logger), osd_id(osd_id),
          notify_osdmap(std::move(notify_osdmap))
    {}

    bool ms_dispatch(Message *m) override
    {
        switch (m->get_type()) {
            case CEPH_MSG_OSD_MAP: {
                auto *osdmap_msg = static_cast<MOSDMap *>(m);
                AGENT_LOG(logger, ::agent::LogLevel::INFO,
                            "received MOSDMap from " +
                                std::string(m->get_source().type_str()) + "." +
                                std::to_string(m->get_source().num()) +
                                " maps=" + std::to_string(osdmap_msg->maps.size()) +
                                " incr=" + std::to_string(osdmap_msg->incremental_maps.size()) +
                                " data_len=" + std::to_string(osdmap_msg->get_data_len()));
                // Distinguish a pull request from a MON-pushed OSDMap:
                // pull requests carry current_epoch in the data bufferlist and
                // have empty maps + incremental_maps.
                if (osdmap_msg->maps.empty() && osdmap_msg->incremental_maps.empty()
                    && osdmap_msg->get_data_len() > 0) {
                    AGENT_LOG(logger, ::agent::LogLevel::INFO,
                                "identified as OSDMap pull request from " +
                                    std::string(m->get_source().type_str()) + "." +
                                    std::to_string(m->get_source().num()));
                    return handle_get_osdmap_request(osdmap_msg);
                }
                return handle_osdmap(osdmap_msg);
            }
            case MSG_OSD_PG_CREATE2:
                return handle_pg_create(static_cast<MOSDPGCreate2 *>(m));
            default:
                AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                            "received unsupported message type=" +
                                std::to_string(m->get_type()));
                return false;
        }
    }

    epoch_t get_last_pg_create_epoch() const
    {
        std::lock_guard l(lock);
        return last_pg_create_epoch;
    }

    void quiesce()
    {
        std::lock_guard l(lock);
        quiesced = true;
    }

    bool ms_handle_reset(Connection *c) override
    {
        AGENT_LOG(logger, ::agent::LogLevel::WARNING,
                    "messenger connection reset from peer=" +
                        c->get_peer_addr().get_legacy_str());
        return false;
    }

    void ms_handle_connect(Connection *c) override
    {
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "outbound connection established to peer=" +
                        c->get_peer_addr().get_legacy_str());
    }

    void ms_handle_accept(Connection *c) override
    {
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "inbound connection accepted from peer=" +
                        c->get_peer_addr().get_legacy_str());
    }

    bool ms_handle_fast_authentication(Connection *con) override
    {
        AGENT_LOG(logger, ::agent::LogLevel::DEBUG,
                    "ms_handle_fast_authentication peer=" +
                        con->get_peer_addr().get_legacy_str() +
                        " peer_type=" + std::to_string(con->get_peer_type()));
        // Accept the connection once the authorizer has been verified by
        // MonClient::handle_auth_request.  Returning false here would cause
        // a valid authorizer to be treated as a failure.
        return true;
    }

    void ms_handle_remote_reset(Connection *c) override
    {
        AGENT_LOG(logger, ::agent::LogLevel::WARNING,
                    "messenger remote reset from peer=" +
                        c->get_peer_addr().get_legacy_str());
    }

    bool ms_handle_refused(Connection *c) override
    {
        AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                    "messenger connection refused peer=" +
                        c->get_peer_addr().get_legacy_str());
        return false;
    }

private:
    bool handle_osdmap(MOSDMap *msg)
    {
        std::lock_guard l(lock);

        if (quiesced) {
            msg->put();
            return true;
        }

        if (msg->fsid != osdmap.get_fsid() && osdmap.get_epoch() != 0) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                        "ignoring OSD map for a foreign fsid");
            msg->put();
            return true;
        }

        for (epoch_t e = osdmap.get_epoch() + 1; e <= msg->get_last(); ++e) {
            auto full = msg->maps.find(e);
            if (full != msg->maps.end()) {
                OSDMap next;
                next.decode(full->second);
                OSDMap::dedup(&osdmap, &next);
                osdmap.deepish_copy_from(next);

                // A full map at epoch e resets the baseline.  Incrementals
                // with epoch < e can no longer form a continuous chain, so
                // trim them from the cache.
                auto it = cached_incrementals.begin();
                while (it != cached_incrementals.end() && it->first < e) {
                    it = cached_incrementals.erase(it);
                }
                continue;
            }

            auto inc = msg->incremental_maps.find(e);
            if (inc != msg->incremental_maps.end() && osdmap.get_epoch() == e - 1) {
                OSDMap::Incremental incremental(inc->second);
                osdmap.apply_incremental(incremental);

                // Cache the raw incremental bufferlist for serving Compute
                // Agent requests with incremental responses.
                cached_incrementals[e] = inc->second;
            }
        }

        // Invalidate the cached full encoding - the local OSDMap has
        // changed, so any previously encoded full map is stale.
        cached_full.epoch = 0;
        cached_full.bl.clear();

        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "received OSD map epoch " +
                        std::to_string(osdmap.get_epoch()) + " up " +
                        std::to_string(osdmap.get_num_up_osds()) + "/" +
                        std::to_string(osdmap.get_num_osds()));
        // We should check osdmap.is_up(osd_id) because if this osd was marked down by
        // monitor, it can still recive osdmap but all the heatbeat(beacon msg) from it
        // will be ignored by monitor. This down OSD MUST resend BOOT message so can it
        // be marked up. This preocess should be finished in DFX story.
        notify_osdmap(osdmap.get_epoch(),
                      osdmap.exists(osd_id) && osdmap.is_up(osd_id));
        msg->put();
        return true;
    }

    bool handle_pg_create(MOSDPGCreate2 *msg)
    {
        std::lock_guard l(lock);

        if (quiesced) {
            msg->put();
            return true;
        }

        last_pg_create_epoch = std::max(last_pg_create_epoch, msg->epoch);

        for (const auto& entry : msg->pgs) {
            const auto& pgid = entry.first;
            created_pgs.insert(pgid);
            monc->send_mon_message(new MOSDPGCreated(pgid.pgid));
            AGENT_LOG(logger, ::agent::LogLevel::INFO,
                        "acknowledged PG create from epoch " +
                            std::to_string(msg->epoch));
        }

        msg->put();
        return true;
    }

    bool handle_get_osdmap_request(MOSDMap *req)
    {
        std::lock_guard l(lock);

        if (quiesced) {
            req->put();
            return true;
        }

        // Decode current_epoch from the data bufferlist.
        epoch_t client_epoch = 0;
        {
            auto p = req->get_data().cbegin();
            ceph::decode(client_epoch, p);
        }
        epoch_t cur_epoch = osdmap.get_epoch();

        auto reply = ceph::make_message<MOSDMap>();
        reply->fsid = osdmap.get_fsid();

        if (cur_epoch == 0 || client_epoch >= cur_epoch) {
            // Client is already up to date (or we have no OSDMap yet).
            reply->newest_map = cur_epoch;

            AGENT_LOG(logger, ::agent::LogLevel::DEBUG,
                        "compute agent up to date at epoch " +
                            std::to_string(client_epoch));
        } else {
            // Try to build an incremental response: check whether every
            // incremental from client_epoch+1 to cur_epoch is cached.
            bool can_incremental = true;
            for (epoch_t e = client_epoch + 1; e <= cur_epoch; ++e) {
                if (cached_incrementals.find(e) == cached_incrementals.end()) {
                    can_incremental = false;
                    break;
                }
            }

            if (can_incremental) {
                for (epoch_t e = client_epoch + 1; e <= cur_epoch; ++e) {
                    reply->incremental_maps[e] = cached_incrementals[e];
                }
                reply->newest_map = cur_epoch;

                AGENT_LOG(logger, ::agent::LogLevel::INFO,
                            "sending incremental OSD map [" +
                                std::to_string(client_epoch + 1) + ".." +
                                std::to_string(cur_epoch) + "] to compute agent");
            } else {
                // Full fallback: send the complete OSDMap encoding.
                if (cached_full.epoch == cur_epoch && cached_full.bl.length() != 0) {
                    // Reuse the cached full encoding.
                    reply->maps[cur_epoch] = cached_full.bl;
                } else {
                    ceph::buffer::list bl;
                    osdmap.encode(bl, CEPH_FEATURES_ALL | CEPH_FEATURE_RESERVED);
                    reply->maps[cur_epoch] = bl;
                    cached_full.epoch = cur_epoch;
                    cached_full.bl = bl;
                }
                reply->newest_map = cur_epoch;

                AGENT_LOG(logger, ::agent::LogLevel::INFO,
                            "sending full OSD map epoch " +
                                std::to_string(cur_epoch) +
                                " to compute agent (client had epoch " +
                                std::to_string(client_epoch) +
                                ", incremental not available)");
            }
        }
        
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "sending OSDMap reply to " +
                        req->get_connection()->get_peer_addr().ip_only_to_str() +
                        " reply_maps=" + std::to_string(reply->maps.size()) +
                        " reply_incremental_maps=" +
                            std::to_string(reply->incremental_maps.size()) +
                        " newest_map=" + std::to_string(reply->newest_map));
        req->get_connection()->send_message2(std::move(reply));
        req->put();
        return true;
    }

    MonClient *monc;
    AgentLogger *logger;
    int osd_id;
    std::function<void(epoch_t, bool)> notify_osdmap;
    mutable ceph::mutex lock = ceph::make_mutex("KVOSDMapDispatcher::lock");
    OSDMap osdmap;
    epoch_t last_pg_create_epoch = 0;
    std::set<spg_t> created_pgs;
    bool quiesced = false;

    // Cached incremental maps received from MON (epoch -> raw buffer).
    // Used to build incremental responses for Compute Agent requests.
    std::map<epoch_t, ceph::buffer::list> cached_incrementals;

    // Cached full OSDMap encoding to avoid repeated encode() calls.
    struct {
        epoch_t epoch = 0;
        ceph::buffer::list bl;
    } cached_full;
};

class KVOSDAgent final : public KVAgentImpl {
public:
    ~KVOSDAgent() override
    {
        shutdown();
        async_worker_stop();
    }

    int init(const KVAgentConfig& config, AgentLogger *agent_logger) override
    {
        logger = agent_logger;
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "initializing OSD agent");
        AGENT_LOG(logger, ::agent::LogLevel::DEBUG,
                    "validating OSD agent configuration");
        if (config.osd_id < 0 || config.mon_host.empty() ||
            config.osd_key.empty() || config.osd_uuid.empty() ||
            config.osd_mon_heartbeat_interval == 0) {
                AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                            "invalid configuration: " + cpp_strerror(-EINVAL));
                return -EINVAL;
            }

        // 保存完整配置副本, 供 mon断链后重建通信层 (messenger/monc) 时复用.
        // 切换 DPU IP 时仅更新 addrs 字段, 其余保持不变.
        saved_config_ = config;
        // 首次 init 时以配置的 addrs 顺序为起点; 重新 init (IP切换) 时保持
        // current_addrs_ (已被 rotate）不变.
        if (!reinitializing_) {
            current_addrs_ = config.addrs;
        }

        {
            AGENT_LOG(logger, ::agent::LogLevel::DEBUG,
                        "acquiring OSD agent state lock");
            std::lock_guard l(state_lock);
            AGENT_LOG(logger, ::agent::LogLevel::DEBUG,
                        "OSD agent state lock acquired");
            if (initialized || starting) {
                AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                            "agent is already initialized: " +
                            cpp_strerror(-EALREADY));
                return -EALREADY;
            }
            starting = true;
            stopping = false;
        }

        CephInitParameters init_params(CEPH_ENTITY_TYPE_OSD);
        init_params.name.set(CEPH_ENTITY_TYPE_OSD,
                             std::to_string(config.osd_id));
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "creating Ceph context for osd." +
                    std::to_string(config.osd_id));
        try {
            cct = boost::intrusive_ptr<CephContext>(
                common_preinit(init_params, CODE_ENVIRONMENT_LIBRARY, 0), false);
        } catch (const std::system_error& error) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                        "failed to create Ceph context: " +
                        std::string(error.what()) + " (" +
                        std::to_string(error.code().value()) + ")");
            std::lock_guard l(state_lock);
            starting = false;
            const int value = error.code().value();
            return value > 0 ? -value : (value < 0 ? value : -EIO);
        }
        int r = cct->_conf.set_val("mon_host", config.mon_host);
        if (r < 0) {
            return fail_init(r, "failed to configure mon_host");
        }
        r = cct->_conf.set_val("key", config.osd_key);
        if (r < 0) {
            return fail_init(r, "failed to configure CephX key");
        }
        r = cct->_conf.set_val("osd_uuid", config.osd_uuid);
        if (r < 0) {
            return fail_init(r, "failed to configure osd_uuid");
        }
        // bind hb source IP to get_myaddrs().front()
        r = cct->_conf.set_val("ms_bind_before_connect", "true");
        if (r < 0) {
            return fail_init(r, "failed to configure ms_bind_before_connect");
        }
        cct->_conf.apply_changes(nullptr);
        common_init_finish(cct.get());
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "Ceph context initialized");
        
        poolctx = std::make_unique<ceph::async::io_context_pool>(
            cct->_conf.get_val<std::uint64_t>("osd_asio_thread_count"));
        poolctx->start(1);
        pool_started = true;

        // 独立于心跳定时器的 MON 连接监控定时器: 无论状态机处于何种状态, 
        // 都周期检查 monc->is_connected(), 一旦发现断链立即触发 DPU IP 切换.
        mon_watch_timer = std::make_unique<boost::asio::steady_timer>(
            poolctx->get_io_context());
        
        monc = std::make_unique<MonClient>(cct.get(), *poolctx);
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "building initial monitor map");
        r = monc->build_initial_monmap();
        if (r < 0) {
            return fail_init(r, "failed to build initial monmap");
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "initial monitor map built");
            
        const auto msg_type = cct->_conf.get_val<std::string>("ms_type");
        // Use nonce 0 so that Compute Agents connecting with the bare
        // address (no nonce) pass the handle_client_ident target_addr check.
        // A random nonce would cause the server to reject connections from
        // clients that don't know our nonce.
        messenger.reset(Messenger::create(
            cct.get(), msg_type, entity_name_t::OSD(config.osd_id), "kv_osd",
            0));
        if (!messenger) {
            return fail_init(-ENOMEM, "failed to create messenger");
        }

        entity_addrvec_t public_addrs;
        r = parse_public_addrs(config.addrs, &public_addrs, logger);
        if (r < 0) {
            return fail_init(r, "failed to parse addrs");
        }

        r = messenger->bindv(public_addrs);
        if (r < 0) {
            return fail_init(r, "failed to bind messenger");
        }

        // 打印 bind 结果, 确认实际绑定的地址和端口
        {
            auto bound = messenger->get_myaddrs();
            std::string addr_str;
            for (size_t i = 0; i < bound.v.size(); ++i) {
                if (i > 0) addr_str += ", ";
                addr_str += bound.v[i].get_legacy_str();
            }
            AGENT_LOG(logger, ::agent::LogLevel::INFO,
                        "messenger bound to [" + addr_str +
                        "] config_addrs=[" + config.addrs + "]");
        }

        // Allow Compute Agents (CEPH_ENTITY_TYPE_CLIENT) to connect and
        // send OSDMap requests.  stateless_server permits arbitrary
        // connect/disconnect/reconnect cycles.
        messenger->set_policy(CEPH_ENTITY_TYPE_CLIENT, 
            Messenger::Policy::stateless_server(0));

        dispatcher = std::make_unique<KVOSDMapDispatcher>(
            cct.get(), monc.get(), logger, config.osd_id,
            [this](epoch_t epoch, bool up) { handle_osdmap(epoch, up); });
        monc->set_messenger(messenger.get());
        // The messenger needs both auth_client (to authenticate ourselves
        // to peers like MON) and auth_server (to validate incoming client
        // connections via cephx).  Without auth_server the accept side
        // faults on the first auth_request frame, resetting the connection.
        messenger->set_auth_client(monc.get());
        messenger->set_auth_server(monc.get());
        messenger->add_dispatcher_tail(dispatcher.get());
        messenger->start();
        messenger_started = true;
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "messenger started, listening for client connections");
        
        monc->set_want_keys(CEPH_ENTITY_TYPE_MON | CEPH_ENTITY_TYPE_OSD);
        // Register the dispatcher so MonClient::handle_auth_request can call
        // ms_handle_fast_authentication after verifying the authorizer.
        monc->set_handle_authentication_dispatcher(dispatcher.get());
        r = monc->init();
        if (r < 0) {
            return fail_init(r, "monclient init failed");
        }
        monc_started = true;

        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "authenticating osd." + std::to_string(config.osd_id) +
                    " (timeout " +
                    std::to_string(KV_AGENT_MON_AUTH_TIMEOUT_SECONDS) +
                    " seconds)");
        r = monc->authenticate(KV_AGENT_MON_AUTH_TIMEOUT_SECONDS);
        if (r < 0) {
            return fail_init(r, "monclient authenticate failed");
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "monitor authentication completed");
        
        // Wait for rotating service keys so we can validate incoming client
        // authorizers.  Without this, the first client connection may arrive
        // before the OSD rotating keys are available, causing the auth_server
        // to reject the authorizer and reset the connection.
        r = monc->wait_auth_rotating(
            KV_AGENT_MON_AUTH_TIMEOUT_SECONDS);
        if (r < 0) {
            return fail_init(r,
                "failed to obtain rotating service keys from monitor");
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "rotating service keys obtained");
        
        superblock.cluster_fsid = monc->get_fsid();
        superblock.osd_fsid = cct->_conf->osd_uuid;
        if (superblock.osd_fsid.is_zero()) {
            return fail_init(-EINVAL, "osd_uuid must be configured");
        }
        superblock.whoami = config.osd_id;
        superblock.current_epoch = 0;
        superblock.weight = 1.0;
        boot_addrs = messenger->get_myaddrs();

        monc->sub_want("osdmap", 0, 0);
        monc->sub_want("osd_pg_creates", dispatcher->get_last_pg_create_epoch(), 0);
        monc->renew_subs();

        heartbeat_interval = config.osd_mon_heartbeat_interval;
        boot_beacon_delay = std::chrono::seconds(config.boot_beacon_delay);
        // 退避参数为编译期宏常量 BOOT_RETRY_DELAY/BOOT_RETRY_MAX_DELAY, 不从 config 读取
        {
            std::lock_guard l(state_lock);
            initialized = true;
            starting = false;
            // First successful init marks the agent as externally initialized
            // stays true even if re-init by ip rotate.
            externally_initialized_ = true;
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "OSD agent initialized");
        return 0;
    }

    bool is_initialized() const override
    {
        std::lock_guard l(state_lock);
        // Report only whether the external caller has performed the first init
        // (externally_initialized_), which is different from the internal initialized flag.
        return externally_initialized_;
    }

    // after MOSDBoot sent, and revieved OSDMap from MON will the hb start.
    int start_heartbeat() override
    {
        std::lock_guard l(state_lock);
        if (!initialized) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                        "agent is not initialized: " + cpp_strerror(-ENOTCONN));
            return -ENOTCONN;
        }
        if (stopping) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                        "agent is stopping: " + cpp_strerror(-ESHUTDOWN));
            return -ESHUTDOWN;
        }
        if (heartbeat_running) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                        "heartbeat is already running: " +
                        cpp_strerror(-EALREADY));
            return -EALREADY;
        }

        heartbeat_running = true;
        // 外部显式 start 即视为恢复, 清除"手动停止"标志 (IP 切换后不再被跳过)
        heartbeat_manually_stopped_ = false;
        heartbeat_timer = std::make_unique<boost::asio::steady_timer>(
            poolctx->get_io_context());
        // 状态机初始化
        state = AgentState::DOWN;
        current_retry_delay = std::chrono::seconds{0};
        // 首次启动 / IP 切换后重启: 都要尽快发 boot.
        // boot_beacon_delay 在切换时被置 0, 保证重新 boot 无 delay.
        if (boot_beacon_delay.count() > 0 && !ip_switch_pending_) {
            arm_boot();
        } else {
            // 直接进入 boot 发送路径 (不等待 boot_beacon_delay)
            send_boot();
        }
        // 启动 MON 连接监控, 检测断链并触发 IP 切换
        mon_was_connected_ = false;
        arm_mon_watch();
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "OSD boot timer started");
        return 0;
    }

    int stop_heartbeat() override
    {
        std::lock_guard l(state_lock);
        if (!initialized) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                        "agent is not initialized: " + cpp_strerror(-ENOTCONN));
            return -ENOTCONN;
        }
        if (stopping) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                        "agent is stopping: " + cpp_strerror(-ESHUTDOWN));
            return -ESHUTDOWN;
        }
        if (!heartbeat_running) {
            // 已处于停止状态，幂等成功
            return 0;
        }
        // cancel触发operation_aborted, handle_heartbeat提前返回且不再加入调度
        heartbeat_timer->cancel();
        heartbeat_running = false;
        state = AgentState::DOWN;
        // 标记手动停止: IP切换重启后不再自动恢复，需外部显式start_heartbeat
        heartbeat_manually_stopped_ = true;
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "OSD heartbeat stopped");
        return 0;
    }

    // 添加 root
    // CRUSH root创建含参数校验、OSDMap获取, CrushWrapper操作等完整流程
    // NOLINTBEGIN(huge_method)
    int32_t kv_agent_create_crush_root_impl(char* kvPoolName, uint32_t nameLen, int32_t* osdIdList, uint32_t listSize) // NOLINT(g.STD.02-cpp)
    {
        if (kvPoolName == nullptr || strnlen(kvPoolName, nameLen + 1) != nameLen) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Invalid kvPoolName");
            return -1;
        }

        // -------- 连接集群 --------
        // --------使用已认证的 monc --------
        // 查询集群中所有已存在的 OSD
        // 打点计时: osd ls 查询集群 OSD 列表耗时
        const auto t_osd_ls_start = std::chrono::steady_clock::now();
        ceph::bufferlist outbl;
        std::string outs;
        int32_t ret = mon_command_sync({R"({"prefix": "osd ls"})"}, &outbl, &outs);
        const auto t_osd_ls_end = std::chrono::steady_clock::now();
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "[TIMING] kv_agent_create_crush_root osd ls elapsed_ms=" +
                    std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                        t_osd_ls_end - t_osd_ls_start).count()));
        if (ret < 0) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to query OSD list: ret=" + std::to_string(ret) + " outs=" + outs);
            return ret;
        }
        std::string osd_list_str = outbl.to_str();
        ret = checkOsdInCluster(osd_list_str, osdIdList, listSize);
        if (ret < 0) {
            return -1;
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO, "All " + std::to_string(listSize) + " OSDs verified to exist in cluster");

        std::string kvPoolNameString = std::string(kvPoolName);
        std::string rootName = kvPoolNameString + "-root";
        std::string ruleName = kvPoolNameString + "-rule";

        // 打点计时: 创建 root bucket 耗时
        const auto t_root_start = std::chrono::steady_clock::now();
        // -------- 1. 创建 root bucket --------
        ret = mon_command_sync({std::string("{\"prefix\": \"osd crush add-bucket\", \"name\": \"") + rootName + "\", \"type\": \"root\"}"}, &outbl, &outs);
        // 已存在判断以 outs 为准, 移出 if (ret < 0):mon 对已存在 bucket 可能返回 0
        // 但 outs 带 "already exists", 仅凭 ret < 0 会漏判.
        bool root_exists = (outs.find("already exists") != std::string::npos ||
                            outs.find("桶已存在") != std::string::npos ||
                            ret == -17);    // EEXIST 兜底

        if (root_exists) {
            AGENT_LOG(logger, ::agent::LogLevel::INFO, "CRUSH root already exists, continuing: " + outs);
        } else if (ret < 0) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to create CRUSH bucket: " + outs);
            return ret;
        } else {
            AGENT_LOG(logger, ::agent::LogLevel::INFO, "CRUSH root created: " + outs);
        }
        // 打点计时: root bucket 创建耗时
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "[TIMING] kv_agent_create_crush_root add-bucket elapsed_ms=" +
                    std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t_root_start).count()));
        // 打点计时: 创建 rule 耗时
        const auto t_rule_start = std::chrono::steady_clock::now();
        // -------- 2. 创建 rule 并绑定到该 root --------
        outbl.clear();
        outs.clear();
        // 命令格式: ceph osd crush rule create-simple <rule-name> <root> <type>
        ret = mon_command_sync({std::string("{\"prefix\": \"osd crush rule create-simple\", \"name\": \"") + ruleName + "\", \"root\": \"" + rootName + "\", \"type\": \"host\"}"}, &outbl, &outs);
        // 已存在判断以 outs 为准, 移出 if (ret < 0), 与 add-bucket 处理保持一致.
        bool rule_exists = (outs.find("already exists") != std::string::npos ||
                            outs.find("规则已存在") != std::string::npos ||
                            ret == -17);    // EEXIST 兜底
        if (rule_exists) {  // 每次都检查一遍, 避免出现 root 创建了 rule 没创建的情况.
            AGENT_LOG(logger, ::agent::LogLevel::INFO, "CRUSH rule already exists, continuing: " + outs);
        } else if (ret < 0) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to create rule: " + outs);
            return ret;
        } else {
            AGENT_LOG(logger, ::agent::LogLevel::INFO, "Rule created: " + outs);
        }
        // 打点计时: rule 创建耗时
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "[TIMING] kv_agent_create_crush_root rule create-simple elapsed_ms=" +
                    std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t_rule_start).count()));
        // 打点计时: moveOsd 将 OSD 移动到 root 耗时
        const auto t_move_start = std::chrono::steady_clock::now();
        // -------- 3. 将 OSD 移动到该 root --------
        ret = moveOsd(rootName, osdIdList, listSize);
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "[TIMING] kv_agent_create_crush_root moveOsd elapsed_ms=" +
                    std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t_move_start).count()));
        if (ret < 0) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to move OSDs to CRUSH bucket: " + outs);
            return ret;
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO, "OSDs moved to root: " + rootName);
        // -------- 清理资源 --------
        outbl.clear();
        outs.clear();
        return 0;
    }
    // NOLINTEND(huge_method)

    // 删除 CRUSH Root 及对应 Rule,  to do: 基于 int32_t* osdIdList, uint32_t listSize 进行缩容, 只有在root为空的情况进行删除。
    // CRUSH Root 删除含多层效验、OSDMap同步、CrushWrapper遍历删除等完整流程
    // NOLINTBEGIN(huge_method, huge_cyclomatic_complexity)
    int32_t kv_agent_delete_crush_root_impl(char* kvPoolName, uint32_t nameLen, int32_t* osdIdList, uint32_t listSize)  // NOLINT(g.STD.02-cpp)
    {
        // -------- 输入合法性检测 --------
        if (kvPoolName == nullptr || strnlen(kvPoolName, nameLen + 1) != nameLen) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Invalid kvPoolName");
            return -1;
        }

        std::string kvPoolNameString = std::string(kvPoolName);
        std::string rootName = kvPoolNameString + "-root";
        std::string ruleName = kvPoolNameString + "-rule";

        // -------- 使用已认证的 monc --------
        ceph::bufferlist outbl;
        std::string outs;

        // -------- 1. 获取 crush dump, 查找 root 及其下的 OSD --------
        outbl.clear();
        outs.clear();
        int32_t ret = mon_command_sync({R"({"prefix": "osd crush dump"})"}, &outbl, &outs);
        if (ret < 0) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to query crush dump: " + outs);
            return ret;
        }

        std::string crush_dump_output = outbl.to_str();

        // -------- 2. 解析 crush dump JSON, 找到目标 root --------
        // 查找 root bucket
        size_t root_name_pos = crush_dump_output.find("\"" + rootName + "\"");
        if (root_name_pos == std::string::npos) {
            AGENT_LOG(logger, ::agent::LogLevel::WARNING, "Root " + rootName + " does not exist in crush map");
            return 0;
        }
        
        // 找到 root 下面的 items 数组, 如果上面存在肯定是能看到的
        size_t items_start = crush_dump_output.find("\"items\"", root_name_pos);
        if (items_start == std::string::npos) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to find items in crush dump for root " + rootName);
            return -1;
        }
        
        // 提取 items 数组
        size_t bracket_start = crush_dump_output.find("[", items_start);
        size_t bracket_end = crush_dump_output.find("]", bracket_start);
        if (bracket_start == std::string::npos || bracket_end == std::string::npos) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to parse items array for root " + rootName);
            return -1;
        }
        std::string items_str = crush_dump_output.substr(bracket_start + 1, bracket_end - bracket_start - 1);

        // 解析 items 中的 OSD ID
        std::vector<int32_t> osds_in_root;
        size_t item_pos = 0;
        while ((item_pos = items_str.find("\"id\"", item_pos)) != std::string::npos) {
            size_t colon_pos = items_str.find(":", item_pos);
            size_t num_start = items_str.find_first_of("0123456789-", colon_pos);
            size_t num_end = items_str.find_first_of(",}]", num_start);
            std::string id_str = items_str.substr(num_start, num_end - num_start);
            try {
                int32_t osd_id = std::stoi(id_str);
                osds_in_root.push_back(osd_id);
            } catch (...) {
                // 忽略无效 ID
            }
            item_pos = num_end;
        }

        {
            std::string msg = "OSDs in root " + rootName + ": ";
            for (size_t i = 0; i < osds_in_root.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += "osd." + std::to_string(osds_in_root[i]);
            }
            AGENT_LOG(logger, ::agent::LogLevel::DEBUG, msg);
        }

        // -------- 3. 验证入参中的 OSD 是否都属于该 root --------
        for (uint32_t i = 0; i < listSize; ++i) {
            int32_t osd_id = osdIdList[i];
            bool found = false;
            for (size_t j = 0; j < osds_in_root.size(); ++j) {
                if (osds_in_root[j] == osd_id) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                AGENT_LOG(logger, ::agent::LogLevel::WARNING, "OSD " + std::to_string(osd_id) + " does not belong to root " + rootName + ", cannot remove");
            }
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO, "All " + std::to_string(listSize) + " OSDs verified");

        // -------- 4. 从 crush map 中移除传入的 OSD --------
        for (uint32_t i = 0; i < listSize; ++i) {
            outbl.clear();
            outs.clear();
            int32_t osd_id = osdIdList[i];
            std::string osd_name = "osd." + std::to_string(osd_id);
            ret = mon_command_sync({std::string("{\"prefix\": \"osd crush remove\", \"name\": \"") + osd_name + "\"}"}, &outbl, &outs);
            if (ret < 0) {
                AGENT_LOG(logger, ::agent::LogLevel::WARNING, "Warning when removing osd." + std::to_string(osd_id) + ": " + outs);
            } else {
                AGENT_LOG(logger, ::agent::LogLevel::INFO, "Try to remove osd." + std::to_string(osd_id) + " from crush map: " + outs);
            }
        }
        
        // -------- 5. 重新获取 crush dump, 检查 root 是否为空 --------
        outbl.clear();
        outs.clear();
        ret = mon_command_sync({R"({"prefix": "osd crush dump"})"}, &outbl, &outs);
        if (ret < 0) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to query crush dump after removal: " + outs);
            return ret;
        }

        crush_dump_output = outbl.to_str();
        root_name_pos = crush_dump_output.find("\"" + rootName + "\"");
        if (root_name_pos == std::string::npos) {
            // root 不存在了, 直接删除 rule
            AGENT_LOG(logger, ::agent::LogLevel::INFO, "Root " + rootName + " no longer exists, proceeding to delete rule");
        } else {
            // 再次检查 items 是否为空
            items_start = crush_dump_output.find("\"items\"", root_name_pos);
            bracket_start = crush_dump_output.find("[", items_start);
            bracket_end = crush_dump_output.find("]", bracket_start);
            std::string items_str_after = crush_dump_output.substr(bracket_start + 1, bracket_end - bracket_start - 1);

            // 检查 items 数组是否为空（只有空白）
            bool root_empty = true;
            for (size_t i = 0; i < items_str_after.length(); ++i) {
                if (items_str_after[i] == '{') {
                    root_empty = false;
                    break;
                }
            }

            if (!root_empty) {
                AGENT_LOG(logger, ::agent::LogLevel::INFO, "Root " + rootName + " is not empty after OSD removal, skipping rule and root deletion");
                return 0;
            }
            AGENT_LOG(logger, ::agent::LogLevel::INFO, "Root " + rootName + " is now empty, proceeding to delete rule and root");
        }

        // -------- 6. 删除 rule --------
        outbl.clear();
        outs.clear();

        ret = mon_command_sync({std::string("{\"prefix\": \"osd crush rule rm\", \"name\": \"") + ruleName + "\"}"}, &outbl, &outs);
        if (ret < 0) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to delete rule: " + outs);
            return ret;
        } else {
            AGENT_LOG(logger, ::agent::LogLevel::INFO, "Rule deleted: " + outs);
        }

        // -------- 7. 删除 root bucket --------
        outbl.clear();
        outs.clear();

        ret = mon_command_sync({std::string("{\"prefix\": \"osd crush remove\", \"name\": \"") + rootName + "\"}"}, &outbl, &outs);
        if (ret < 0) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to delete root bucket: " + outs);
            return ret;
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO, "Root bucket deleted: " + outs);

        // -------- 清理资源 --------
        return 0;
    }
    // NOLINTEND(huge_method, huge_cyclomatic_complexity)

    // 创建cephpool
    int32_t kv_agent_create_ceph_pool_impl(char* kvPoolName, uint32_t kvPoolLen, char* kvnsName, uint32_t kvnsLen)  // NOLINT(g.STD.02-cpp)
    {
        // -------- 输入合法性检测 --------
        if (kvnsName == nullptr || strnlen(kvnsName, kvnsLen + 1) != kvnsLen) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Invalid kvnsName");
            return -1;
        }
        if (kvPoolName == nullptr || strnlen(kvPoolName, kvPoolLen + 1) != kvPoolLen) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Invalid kvPoolName");
            return -1;
        }
        // -------- 使用已认证的 monc --------
        std::string kvPoolNameString = std::string(kvPoolName);
        std::string kvnsNameString = std::string(kvnsName);
        std::string ruleName = kvPoolNameString + "-rule";

        ceph::bufferlist outbl;
        std::string outs;
        // 打点计时: 创建 ceph pool 耗时
        const auto t_pool_start = std::chrono::steady_clock::now();
        // -------- 创建 ceph pool --------
        // 命令格式: ceph osd pool create <pool> [<pg_num:int>] [<pgp_num:int>] [<pool_type:replicated|erasure>] [<rule>]
        // 创建ceph pool并绑定对应的规则
        int32_t ret = mon_command_sync({std::string("{\"prefix\": \"osd pool create\", \"pool\": \"") + kvnsNameString + "\", \"pg_num\": 4, \"pool_type\": \"replicated\", \"rule\": \"" + ruleName + "\"}"}, &outbl, &outs);
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                  "kv_agent_create_ceph_pool: osd pool create ret=" +
                  std::to_string(ret) + ", pool=" + kvnsNameString +
                  ", rule=" + ruleName + ", outs=" + outs);
        // 判断"已存在"以 outs 内容为准, 不依赖 ret < 0: mon 对已存在的 pool 执行
        // create 可能返回 0 但 outs 带 "already exists" (见日志 ret = 0, outs = pool 'kvns'
        // already exists 场景), 故将其移出 if (ret < 0).
        bool ceph_pool_exists = (outs.find("already exists") != std::string::npos ||
                                    outs.find("桶已存在") != std::string::npos ||
                                    ret == -17);    // EEXIST 兜底

        if (ceph_pool_exists) {
            AGENT_LOG(logger, ::agent::LogLevel::INFO, "ceph pool already exists, continuing: " + outs);
            outbl.clear();
            outs.clear();
            // ceph osd pool get <pool> crush rule -> "crush_rule: <rule-name>"
            int32_t getRuleRet = mon_command_sync({std::string("{\"prefix\": \"osd pool get\", \"pool\": \"") + kvnsNameString + "\", \"var\": \"crush_rule\"}"}, &outbl, &outs);
            if (getRuleRet < 0) {
                AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to query crush rule of pool " + kvnsNameString + ": " + outs);
                return getRuleRet;
            }
            // 解析返回的 rule 名: 取最后一个 ':' 之后的字段并去除空白
            std::string ruleOut = outbl.to_str();
            std::string boundRule;
            size_t colonPos = ruleOut.rfind(':');
            if (colonPos != std::string::npos) {
                boundRule = ruleOut.substr(colonPos + 1);
            }
            size_t leadPos = boundRule.find_first_not_of(" \t\r\n");
            if (leadPos == std::string::npos) {
                boundRule.clear();
            } else {
                boundRule = boundRule.substr(leadPos);
                size_t tailPos = boundRule.find_last_not_of(" \t\r\n");
                if (tailPos != std::string::npos) {
                    boundRule = boundRule.substr(0, tailPos + 1);
                }
            }
            if (boundRule != ruleName) {
                AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                          "ceph pool " + kvnsNameString + " already bound to another crush rule, expected: " +
                          ruleName + ", actual: " + boundRule);
                return -17; // EEXIST: pool 已存在但绑定 rule 与期望不一致
            }
            AGENT_LOG(logger, ::agent::LogLevel::INFO, "ceph pool " + kvnsNameString + " bound to expected crush rule: " + ruleName);
        } else if (ret < 0) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to create ceph pool: " + outs);
            return ret;
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO, "Ceph pool created: " + outs);
        // 打点计时: ceph pool 创建耗时
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "[TIMING] kv_agent_create_ceph_pool pool create elapsed_ms=" +
                    std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t_pool_start).count()));
        return 0;
    }

    // 删除ceph pool
    int32_t kv_agent_delete_ceph_pool_impl(char* kvnsName, uint32_t nameLen) // NOLINT(g.STD.02-cpp)
    {
        // -------- 输入合法性检测 --------
        if (kvnsName == nullptr || strnlen(kvnsName, nameLen + 1) != nameLen) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Invalid kvnsName");
            return -1;
        }
        // -------- 使用已认证的 monc --------
        std::string kvnsNameString = std::string(kvnsName);
    
        ceph::bufferlist outbl;
        std::string outs;

        // -------- 检查 ceph pool 是否存在 --------
        outbl.clear();
        outs.clear();
        int32_t ret = mon_command_sync({R"({"prefix": "osd pool ls"})"}, &outbl, &outs);
        if (ret < 0) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to query pool list: " + outs);
            return ret;
        }

        std::string pool_list_str = outbl.to_str();
        if (pool_list_str.find("\"" + kvnsNameString + "\"") == std::string::npos &&
            pool_list_str.find(kvnsNameString) == std::string::npos) {
            AGENT_LOG(logger, ::agent::LogLevel::WARNING, "Ceph pool " + kvnsNameString + " does not exist");
            return 0;
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO, "Ceph pool " + kvnsNameString + " exists, proceeding to delete");

        // -------- 删除 ceph pool --------
        // 命令格式: ceph osd pool delete <pool-name> <pool-name> --yes-i-really-really-mean-it
        outbl.clear();
        outs.clear();
        ret = mon_command_sync({std::string("{\"prefix\": \"osd pool delete\", \"pool\": \"") + kvnsNameString + "\", \"pool2\": \"" + kvnsNameString + "\", \"yes_i_really_really_mean_it\": true}"}, &outbl, &outs);
        if (ret < 0) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to delete ceph pool: " + outs);
            return ret;
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO, "Ceph pool deleted: " + outs);
        return 0;
    }

    // ============== 异步变体 ==============
    // 4个 _async 方法将任务投递到单 worker 线程, 在 worker 内执行对应sync
    // 实现, 完成后调用回调. 单 worker 足够: 4个 crush/pool 操作共享
    // g_crush_mutex 串行执行, 多 worker 也不会并发, 反而增加资源开销.
    int32_t kv_agent_create_crush_root_async(char* kvPoolName, uint32_t nameLen,
        int32_t* osdIdList, uint32_t listSize,
        KVAgentAsyncCallback cb, void *user_data) override
    {
        return async_dispatch([this, kvPoolName, nameLen, osdIdList, listSize]() {
            return kv_agent_create_crush_root_impl(kvPoolName, nameLen, osdIdList, listSize);
        }, std::move(cb), user_data);
    }

    int32_t kv_agent_delete_crush_root_async(char* kvPoolName, uint32_t nameLen,
        int32_t* osdIdList, uint32_t listSize,
        KVAgentAsyncCallback cb, void *user_data) override
    {
        return async_dispatch([this, kvPoolName, nameLen, osdIdList, listSize]() {
            return kv_agent_delete_crush_root_impl(kvPoolName, nameLen, osdIdList, listSize);
        }, std::move(cb), user_data);
    }

    int32_t kv_agent_create_ceph_pool_async(char* kvPoolName, uint32_t kvPoolLen,
        char* kvnsName, uint32_t kvnsLen,
        KVAgentAsyncCallback cb, void *user_data) override
    {
        return async_dispatch([this, kvPoolName, kvPoolLen, kvnsName, kvnsLen]() {
            return kv_agent_create_ceph_pool_impl(kvPoolName, kvPoolLen, kvnsName, kvnsLen);
        }, std::move(cb), user_data);
    }

    int32_t kv_agent_delete_ceph_pool_async(char* kvnsName, uint32_t nameLen,
        KVAgentAsyncCallback cb, void *user_data) override
    {
        return async_dispatch([this, kvnsName, nameLen]() {
            return kv_agent_delete_ceph_pool_impl(kvnsName, nameLen);
        }, std::move(cb), user_data);
    }

private:
    // ============== async worker thread pool ==============
    // 单 worker 线程 + FIFO 任务队列. 任务在 worker 线程内执行 sync 实现,
    // 完成后调用回调. 调用方 (service 侧) 通过回调里的 LWT 信号量 UP 唤醒
    // LWT 任务, 避免 LWT 任务直接 pthread_cond_wait 阻塞 OS 线程.
    struct AsyncTask {
        std::function<int32_t()> fn;    // 包装的 sync 调用
        KVAgentAsyncCallback cb;        // 完成回调
        void *user_data;                // 透传指针
    };

    std::mutex async_mtx_;
    std::condition_variable async_cv_;
    std::queue<AsyncTask> async_queue_;
    std::thread async_worker_;
    bool async_worker_started_ = false;
    bool async_stop_ = false;

    void async_worker_start()
    {
        std::lock_guard<std::mutex> lock(async_mtx_);
        if (async_worker_started_) {
            return;
        }
        async_stop_ = false;
        async_worker_ = std::thread([this] { async_worker_loop(); });
        async_worker_started_ = true;
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                  "async worker thread started");
    }

    void async_worker_stop()
    {
        {
            std::lock_guard<std::mutex> lock(async_mtx_);
            if (!async_worker_started_) {
                return;
            }
            async_stop_ = true;
            async_cv_.notify_all();
        }
        if (async_worker_.joinable()) {
            async_worker_.join();
        }
        {
            std::lock_guard<std::mutex> lock(async_mtx_);
            async_worker_started_ = false;
            std::queue<AsyncTask> empty;
            async_queue_.swap(empty);
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                  "async worker thread stopped");
    }

    void async_worker_loop()
    {
        for (;;) {
            AsyncTask task;
            {
                std::unique_lock<std::mutex> lock(async_mtx_);
                async_cv_.wait(lock, [this] {
                    return async_stop_ || !async_queue_.empty();
                });
                if (async_stop_ && async_queue_.empty()) {
                    return;
                }
                task = std::move(async_queue_.front());
                async_queue_.pop();
            }
            int32_t result = -EINVAL;
            try {
                result = task.fn();
            } catch (const std::exception& e) {
                AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                          std::string("async task exception: ") + e.what());
                result = -EIO;
            } catch (...) {
                AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                          "async task unknown exception");
                result = -EIO;
            }
            try {
                if (task.cb) {
                    task.cb(result, task.user_data);
                }
            } catch (...) {
                AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                          "async callback exception swallowed");
            }
        }
    }

    // 投递任务到 worker. cb/user_data 在 worker 完成后被调用.
    // 返回 0 表示已投递; 负数表示失败 (worker 未启动 / 投递异常).
    int32_t async_dispatch(std::function<int32_t()> fn,
                           KVAgentAsyncCallback cb, void *user_data)
    {
        if (!cb) {
            AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                      "async_dispatch: callback is null");
            return -EINVAL;
        }
        async_worker_start();
        {
            std::lock_guard<std::mutex> lock(async_mtx_);
            if (async_stop_) {
                AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                          "async_dispatch: worker is stopping");
                return -ESHUTDOWN;
            }
            async_queue_.push(AsyncTask{std::move(fn), std::move(cb), user_data});
        }
        async_cv_.notify_one();
        return 0;
    }

private:
    // ---- DPU IP 切换: MON 连接监控 ----
    // 独立于状态机的定时器, 周期检查 monc->is_connected().
    // 一旦从 connected 变为 disconnected, 说明当前 DPU 心跳 IP (myaddrs 第一
    // 个) 不可达, 立即切换下一个 IP 并重新走 BOOT 流程
    void arm_mon_watch()
    {
        if (!mon_watch_timer) {
            return;
        }
        // 断链轮询间隔与心跳间隔 (osd_mon_heartbeat_interval) 保持一致: 
        // 正常情况下每心跳周期检查一次 MON 连接; 断链切换时也按此节奏轮询.
        mon_watch_timer->expires_after(
            std::chrono::seconds(std::max(1, heartbeat_interval)));
        mon_watch_timer->async_wait(
            [this](const boost::system::error_code& error) {
                handle_mon_watch(error);
            });
    }

    void handle_mon_watch(const boost::system::error_code& error)
    {
        if (error == boost::asio::error::operation_aborted || stopping ||
            !initialized) {
            return;
        }
        if (error || !heartbeat_running) {
            return;
        }

        bool need_restart = false;
        {
            std::lock_guard l(state_lock);
            if (monc && monc->is_connected()) {
                mon_was_connected_ = true;
            } else if (mon_was_connected_ && !ip_switch_pending_) {
                // 曾经连上过 MON, 现在断了: 当前心跳 IP 不可达, 触发切换
                mon_was_connected_ = false;
                ip_switch_pending_ = true;
                need_restart = true;
            }
        }

        if (need_restart) {
            AGENT_LOG(logger, ::agent::LogLevel::WARNING,
                        "MON connection lost, switching DPU IP and rebooting");
            // 完整重建 (shutdown + re-init) 会 stop 并销毁当前 io_context
            // (poolctx). 不能在本回调线程 (poolctx 的 io_context 线程) 内销毁
            // 自己所在的 io_context, 因此切换到独立线程执行重启, 由重启后的
            // start_heartbeat 重新 arm_mon_watch.
            std::thread restart_thread([this] {
                restart_for_ip_switch();
                std::lock_guard l(state_lock);
                ip_switch_pending_ = false;
            });
            restart_thread.detach();
            return;
        }

        arm_mon_watch();
    }

    // 将当前 addrs 字符串中的第一个 IP 移到末尾, 使第二个 IP 成为第一个.
    // 返回 false 表示没有可切换的 IP  (列表长度 < 2).
    bool rotate_addrs()
    {
        std::vector<std::string> ips;
        size_t begin = 0;
        while (begin < current_addrs_.size()) {
            const auto comma = current_addrs_.find(',', begin);
            ips.push_back(current_addrs_.substr(begin, comma - begin));
            if (comma == std::string::npos) {
                break;
            }
            begin = comma + 1;
        }
        if (ips.size() < 2) {
            AGENT_LOG(logger, ::agent::LogLevel::WARNING,
                        "cannot switch DPU IP: only " + 
                        std::to_string(ips.size()) + " addr(s) configured");
            return false;
        }
        const std::string first = ips.front();
        ips.erase(ips.begin());
        ips.push_back(first);
        std::string new_addrs;
        for (size_t i = 0; i < ips.size(); ++i) {
            if (i > 0) {
                new_addrs += ",";
            }
            new_addrs += ips[i];
        }
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "rotating DPU addrs: " + current_addrs_ + " -> " + new_addrs);
        current_addrs_ = new_addrs;
        return true;
    }

    // 销毁 messenger/monc 等通信层, 用新顺序 addrs 重新走 BOOT 流程.
    // 重新 init 时 bindv(新顺序 addrs) -> _finish_bind -> set_myaddrs(新顺序),
    // 因此出站心跳源 IP (get_myaddrs().front()) 与 client_ident 上报顺序
    // 都会切换到新 IP, MON 更新 osdmap 后 client agent 自动连新 IP.
    // 调用前 current_addrs_ 必须已被 rotate 为新顺序 (见 handle_mon_watch).
    // init() 是单次尝试 (失败返回错误码), 此处循环尝试每个候选 IP 直到成功:
    // 首个候选 IP 认证超时失败后 rotate 到下一个 IP 重新 init, 所有 IP 都
    // 不可达时也按心跳间隔轮询等待, 待 MON 恢复后自动建立连接.
    void restart_for_ip_switch()
    {
        // 保存当前 logger 引用 (shutdown 后仍有效, logger 由外部持有)
        AgentLogger *saved_logger = logger;
        // 重建配置: 仅替换 addrs 为当前顺序 (首个 IP 为候选心跳 IP)
        KVAgentConfig cfg = saved_config_;

        // 当前第一个 IP 已断链 (触发本次切换), 先 rotate 把它移到末尾,
        // 让下一个 IP 成为候选, 然后循环尝试直到成功.
        if (!rotate_addrs()) {
            AGENT_LOG(saved_logger, ::agent::LogLevel::WARNING,
                        "no alternate DPU IP to switch to, retrying same addrs");
        }

        while (true) {
            cfg.addrs = current_addrs_;

            shutdown();

            // 重新 init 时不重置 current_addrs_ (保留已 rotate 的顺序)
            reinitializing_ = true;
            const int r = init(cfg, saved_logger);
            reinitializing_ = false;
            if (r < 0) {
                AGENT_LOG(saved_logger, ::agent::LogLevel::ERROR,
                            "re-init with addrs " + current_addrs_ + " failed: " + 
                            cpp_strerror(r) + ", rotating to next IP");
                // 多 IP 时已 rotate 到下一个候选; 单 IP 时 rotate 返回 false, 但
                // 仍按心跳间隔等待 MON 恢复后重试, 保证 agent 不会进入死状态.
                rotate_addrs();
                std::this_thread::sleep_for(
                    std::chrono::seconds(std::max(1, heartbeat_interval)));
                continue;
            }

            // init 成功: 重新走 BOOT 流程, 不等待 boot_beacon_delay
            boot_beacon_delay = std::chrono::seconds{0};
            // 若曾主动停止发心跳 (stop_heartbeat), IP 切换重启后保持静默,
            // 不自动恢复发心跳, 需外部显式 start_heartbeat() 恢复
            if (heartbeat_manually_stopped_) {
                AGENT_LOG(saved_logger, ::agent::LogLevel::INFO,
                            "IP switch finished but heartbeat was manually stopped, "
                            "staying silent until kv_agent_start_heartbeat");
                return;
            }
            const int hb = start_heartbeat();
            if (hb < 0) {
                AGENT_LOG(saved_logger, ::agent::LogLevel::ERROR,
                            "restart heartbeat after IP switch failed: " + 
                            cpp_strerror(hb));
            }
            return;
        }
    }

    void arm_boot()
    {
        heartbeat_timer->expires_after(boot_beacon_delay);
        heartbeat_timer->async_wait(
            [this](const boost::system::error_code& error) {
                handle_boot_timer(error);
            });
    }

    void handle_boot_timer(const boost::system::error_code& error)
    {
        // expires_after 取消旧定时器时收到 operation_aborted, 不是错误, 不重置 heartbeat_running
        if (error == boost::asio::error::operation_aborted || stopping || !initialized) {
            return;
        }
        if (error || !heartbeat_running) {
            heartbeat_running = false;
            return;
        }
        std::lock_guard l(state_lock);
        // 仅在 DOWN 状态下处理 boot 定时器到期
        if (state != AgentState::DOWN) {
            return;
        }
        send_boot();
    }

    // 统一 MOSDBoot 发送入口, 调用者必须已持有 state_lock。
    // 守卫: stopping/未初始化/心跳未运行时不发送。
    // 哨兵值0表示首次发送, 初始化为 BOOT_RETRY_DELAY（编译期宏常量）。
    void send_boot()
    {
        if (stopping || !initialized || !heartbeat_running) {
            return;
        }

        // 同步 OSDMap epoch 到 superblock.current_epoch 与 MOSDBoot.version, 
        // 两者必须与 MON 当前 osdmap.get_epoch() 一致才能避免 boot 死循环:
        // - MOSDBoot.version (构造函数第二参数 e): MON preprocess_boot 用它与
        //   up_from 比较, 若 version < up_from 则判为"过时 boot", 只回推 OSDMap
        //   不标记 up; 当 version + 1 > mon epoch 时 send_incremental 空循环不回包,
        //   导致 agent 永远收不到回复. 原生 OSD (OSD.cc:_send_boot) 传
        //   get_osdmap_epoch().
        // - sb.current_epoch: MON send_latest 的起始 epoch, 用于回推增量 OSDMap.
        superblock.current_epoch = osdmap_epoch;

        auto boot = new MOSDBoot(superblock, osdmap_epoch, 0, boot_addrs, boot_addrs,
                                boot_addrs, CEPH_FEATURES_ALL);
        boot->metadata["kv_node"] = "true";
        boot->metadata["agent"] = "kv-osd";
        boot->metadata["osd_id"] = std::to_string(superblock.whoami);

        // 打印上报给 MON 的 boot_addrs, 确认 OSDMap 中的地址与实际 bind 地址一致
        {
            std::string addr_str;
            for (size_t i = 0; i < boot_addrs.v.size(); ++i) {
                if (i > 0) addr_str += ", ";
                addr_str += boot_addrs.v[i].get_legacy_str();
            }
            AGENT_LOG(logger, ::agent::LogLevel::INFO,
                        "send_boot: osd_id=" + std::to_string(superblock.whoami) + 
                        " epoch=" + std::to_string(osdmap_epoch) +
                        " boot_addrs=[" + addr_str + "]");
        }

        monc->send_mon_message(boot);
        state = AgentState::BOOT_SENT;
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "sent OSD boot message");
                
        if (current_retry_delay.count() == 0) {
            current_retry_delay = std::chrono::seconds(BOOT_RETRY_DELAY);
        }

        arm_boot_retry();
    }

    // 装 retry 定时器, 调用者必须已持有 state_lock.
    // 复用 heartbeat_timer: expires_after 自动取消旧 pending.
    void arm_boot_retry()
    {
        heartbeat_timer->expires_after(current_retry_delay);
        heartbeat_timer->async_wait(
            [this](const boost::system::error_code& error) {
                handle_boot_retry(error);
            });
    }

    // retry 定时器回调, 指数退避.
    // state 已被 OSDMap 转移时静默返回; 否则翻倍 current_retry_delay
    // (封顶 BOOT_RETRY_MAX_DELAY) 并重发 boot.
    void handle_boot_retry(const boost::system::error_code& error)
    {
        // expires_after 取消旧定时器时收到 operation_aborted, 不是错误, 不重置 heartbeat_running
        if (error == boost::asio::error::operation_aborted || stopping || !initialized) {
            return;
        }
        if (error || !heartbeat_running) {
            heartbeat_running = false;
            return;
        }

        std::lock_guard l(state_lock);
        
        if (state != AgentState::BOOT_SENT) {
            return;
        }

        current_retry_delay *= 2;

        if (current_retry_delay > std::chrono::seconds(BOOT_RETRY_MAX_DELAY)) {
            current_retry_delay = std::chrono::seconds(BOOT_RETRY_MAX_DELAY);
        }

        send_boot();
    }

    void handle_osdmap(epoch_t epoch, bool up)
    {
        std::lock_guard l(state_lock);

        // 更新 epoch
        osdmap_epoch = epoch;

        // 守卫: stopping/未初始化/心跳未运行时不处理
        if (stopping || !initialized || !heartbeat_running) {
            return;
        }

        switch (state) {
        case AgentState::DOWN:
            // DOWN 状态: 无论 up=true 还是 up=false 都忽略
            // up=true: 未发 boot, 不期望被标 up
            // up=false: 维持 DOWN
            break;
        
        case AgentState::BOOT_SENT:
            if (up) {
                // BOOT_SENT + up=true -> ACTIVE
                state = AgentState::ACTIVE;
                // cancel retry 定时器 (expires_after 自动取消旧 pending)
                // 重置退避
                current_retry_delay = std::chrono::seconds{0};
                // 发送首个 beacon 并装填心跳定时器
                send_heartbeat(epoch);
                arm_heartbeat();
            }
            // BOOT_SENT + up=false -> 维持 BOOT_SENT (MON 尚未处理 boot, 继续等 retry)
            break;
        
        case AgentState::ACTIVE:
            if (up) {
                // ACTIVE + up=true -> 维持 ACTIVE, epoch 已更新
            } else {
                // ACTIVE + up=false -> DOWN
                state = AgentState::DOWN;
                // cancel 心跳定时器 (通过后续 send_boot->arm_boot_retry 覆盖)
                // 重置退避
                current_retry_delay = std::chrono::seconds{0};
                // 立即 send_boot (不等 boot_beacon_delay, KVService 已在运行)
                send_boot();
            }
            break;
        }
    }

    void arm_heartbeat()
    {
        heartbeat_timer->expires_after(
            std::chrono::seconds(std::max(1, heartbeat_interval)) +
            std::chrono::milliseconds(jitter_dist(rng)));
        heartbeat_timer->async_wait(
            [this](const boost::system::error_code& error) {
                handle_heartbeat(error);
            });
    }

    void send_heartbeat(epoch_t epoch)
    {
        monc->send_mon_message(
            new MOSDBeacon(epoch, 0, utime_t(), heartbeat_interval));
    }

    void handle_heartbeat(const boost::system::error_code& error)
    {
        // expires_after 取消旧定时器时收到 operation_aborted, 不是错误, 不重置 heartbeat_running
        if (error == boost::asio::error::operation_aborted || stopping || !initialized) {
            return;
        }
        if (error) {
            heartbeat_running = false;
            return;
        }
        std::lock_guard l(state_lock);
        // 仅在 ACTIVE 状态下发送心跳
        if (state != AgentState::ACTIVE) {
            return;
        }
        send_heartbeat(osdmap_epoch);
        arm_heartbeat();
    }

    void request_stop()
    {
        std::lock_guard l(state_lock);
        stopping = true;
        if (heartbeat_timer) {
            heartbeat_timer->cancel();
        }
    }

    int fail_init(int r, const char *message)
    {
        AGENT_LOG(logger, ::agent::LogLevel::ERROR,
                    std::string(message) + ": " + cpp_strerror(r));
        shutdown();
        std::lock_guard l(state_lock);
        starting = false;
        return r;
    }

    void shutdown()
    {
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "stopping OSD agent");
        request_stop();
        std::lock_guard shutdown_guard(shutdown_lock);

        if (mon_watch_timer) {
            mon_watch_timer->cancel();
        }
        if (dispatcher) {
            dispatcher->quiesce();
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
        if (pool_started) {
            poolctx->stop();
            pool_started = false;
        }

        heartbeat_timer.reset();
        mon_watch_timer.reset();
        dispatcher.reset();
        messenger.reset();
        monc.reset();
        poolctx.reset();
        cct.reset();

        std::lock_guard l(state_lock);
        initialized = false;
        heartbeat_running = false;
        state = AgentState::DOWN;
        current_retry_delay = std::chrono::seconds{0};
        osdmap_epoch = 0;
        AGENT_LOG(logger, ::agent::LogLevel::INFO,
                    "OSD agent stopped");
    }


    // 同步封装 monc->start_mon_command
    struct SyncContext : public Context {
        std::mutex& mtx;
        std::condition_variable& cv;
        bool& done;
        int& result;
        std::string& result_outs;
        std::string* outs;

        SyncContext(std::mutex& m, std::condition_variable& c, bool& d,
                    int& r, std::string& ru, std::string* o)
            : mtx(m), cv(c), done(d), result(r), result_outs(ru), outs(o) {}

        void finish(int r) override {
            std::lock_guard<std::mutex> lock(mtx);
            result = r;
            if (outs) *outs = result_outs;
            done = true;
            cv.notify_one();
        }
    };

    int32_t mon_command_sync(std::vector<std::string> cmd,
                             ceph::bufferlist *outbl,
                             std::string *outs)
    {
        ceph::bufferlist inbl;
        std::mutex mtx;
        std::condition_variable cv;
        bool done = false;
        int result = 0;
        std::string result_outs;

        monc->start_mon_command(
            std::move(cmd), std::move(inbl), outbl, &result_outs,
            new SyncContext(mtx, cv, done, result, result_outs, outs));

        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&done] { return done; });
        return (result < 0) ? result : 0;
    }

    int32_t moveOsd(std::string rootName, int32_t* osdIdList, uint32_t listSize)
    {
        ceph::bufferlist outbl;
        std::string outs;

        for (uint32_t i = 0; i < listSize; ++i) {
            int32_t osd_id = osdIdList[i];

            // 使用 osd crush add 添加 OSD 到 CRUSH map
            int32_t ret = mon_command_sync({std::string("{\"prefix\": \"osd crush add\", \"id\": ") + std::to_string(osd_id) + ", \"weight\": 1.0, \"args\": [\"root=" + rootName + "\"]}"}, &outbl, &outs);

            if (ret < 0) {
                AGENT_LOG(logger, ::agent::LogLevel::ERROR, "Failed to add osd." + std::to_string(osd_id) + " to root " + rootName + ": " + outs);
                return ret;
            } else {
                AGENT_LOG(logger, ::agent::LogLevel::INFO, "Successfully added osd." + std::to_string(osd_id) + " to root " + rootName + " (via add)");
            }

            // 清理, 为下一次使用准备
            outbl.clear();
            outs.clear();
        }

        return 0;
    }


    // 将查询结果字符串转变为osd列表
    std::vector<int32_t> getOsds(std::string osd_list_str)
    {
        std::vector<int32_t> existing_osds;
        size_t pos = 0;
        while (pos < osd_list_str.length()) {
            size_t newline_pos = osd_list_str.find('\n', pos);
            if (newline_pos == std::string::npos) {
                newline_pos = osd_list_str.length();
            }
            std::string osd_id_str = osd_list_str.substr(pos, newline_pos - pos);
            // 跳过空行和空白
            if (!osd_id_str.empty() && osd_id_str.find_first_not_of(" \t\r\n") != std::string::npos) {
                try {
                    int32_t osd_id = std::stoi(osd_id_str);
                    existing_osds.push_back(osd_id);
                } catch (...) {
                    // 忽略无效的 OSD ID
                }
            }
            pos = newline_pos + 1;
        }
        return existing_osds;
    }

    // 确认列表中的osd是否都在集群中
    int32_t checkOsdInCluster(std::string osd_list_str, int32_t* osdIdList, uint32_t listSize)
    {
        std::vector<int32_t> existing_osds = getOsds(osd_list_str);

        // 检查请求的每个 OSD 是否存在
        for (uint32_t i = 0; i < listSize; ++i) {
            int32_t osd_id = osdIdList[i];
            bool found = false;
            for (size_t j = 0; j < existing_osds.size(); ++j) {
                if (existing_osds[j] == osd_id) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                AGENT_LOG(logger, ::agent::LogLevel::ERROR, "OSD " + std::to_string(osd_id) + " does not exist in cluster");
                return -1;
            }
        }
        return 0;
    }

    boost::intrusive_ptr<CephContext> cct;
    std::unique_ptr<ceph::async::io_context_pool> poolctx;
    std::unique_ptr<MonClient> monc;
    std::unique_ptr<Messenger> messenger;
    std::unique_ptr<KVOSDMapDispatcher> dispatcher;
    std::unique_ptr<boost::asio::steady_timer> heartbeat_timer;
    OSDSuperblock superblock;
    entity_addrvec_t boot_addrs;

    mutable std::mutex state_lock;
    std::mutex shutdown_lock;
    bool starting = false;
    // internal state. Will briefly becomes false during the IP-switch rebuild
    // (shutdown+re-init) window
    bool initialized = false;
    // Flag for whether the external caller has performed the first init: set to
    // true on the first successful init and never reset.
    bool externally_initialized_ = false;
    bool heartbeat_running = false;
    // 主动停止发心跳 (stop_heartbeat) 标记: 为 true 时 IP 切换重启后不自动
    // 恢复发心跳, 需外部显式 start_heartbeat() 恢复
    bool heartbeat_manually_stopped_ = false;
    AgentState state = AgentState::DOWN;
    std::chrono::seconds current_retry_delay{0};    // 0 = 哨兵值, send_boot 中初始化为 BOOT_RETRY_DELAY
    bool stopping = false;
    bool pool_started = false;
    bool messenger_started = false;
    bool monc_started = false;
    int heartbeat_interval = 1;
    static constexpr int jitter_ms_range = 1000;    // 心跳随机抖动时间, 毫秒为单位
    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> jitter_dist{-jitter_ms_range, jitter_ms_range};
    epoch_t osdmap_epoch = 0;
    std::chrono::seconds boot_beacon_delay{0};
    AgentLogger *logger = nullptr;

    // ---- DPU IP 切换 feature 状态 ----
    KVAgentConfig saved_config_;                // init时保存的完整配置副本
    std::string current_addrs_;                 // 当前 addrs 字符串 (可被 rotate)
    std::unique_ptr<boost::asio::steady_timer> mon_watch_timer; // MON断链监控定时器
    bool mon_was_connected_ = false;            // 上一次 monc->is_connected() 结果
    bool ip_switch_pending_ = false;            // 是否刚完成 IP 切换 (重启后立即 boot)
    bool reinitializing_ = false;               // 是否处于重新 init 流程 (保留 current_addrs_)
};

} // namespace

} // namespace agent

namespace agent {

std::unique_ptr<KVAgentImpl> create_kv_osd_agent()
{
    return std::make_unique<KVOSDAgent>();
}

} // namespace agent
