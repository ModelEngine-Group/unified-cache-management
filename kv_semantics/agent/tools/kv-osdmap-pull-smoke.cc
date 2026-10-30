// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#include "KVAgentLogger.h"

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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

namespace {

class OSDPullAgent final : public Dispatcher {
public:
    OSDPullAgent()
     : Dispatcher(nullptr)
    {}

    ~OSDPullAgent() override
    {
        shutdown();
    }

    int init(const std::string& mon_host,
             const std::string& client_name,
             const std::string& client_key,
             agent::AgentLogger *agent_logger)
    {
        logger = agent_logger;
        std::cerr << "[trace] init: ENTER - creating CephContext" << std::endl;

        CephInitParameters params(CEPH_ENTITY_TYPE_CLIENT);
        if (!params.name.from_str(client_name)) {
            std::cerr << "[trace] init: failed to parse client_name='" <<
                      << client_name << "'" << std::endl;
            return -EINVAL;
        }
        context = boost::intrusive_ptr<CephContext>(
            common_preinit(params, CODE_ENVIRONMENT_LIBRARY, 0), false);
        std::cerr << "[trace] init: CephContest created" << std::endl;
        int r = context->_conf.set_val("mon_host", mon_host);
        if (r < 0) {
            std::cerr << "[trace] init: set_val(mon_host) failed r=" << r << std::endl;
            return r;
        }
        r =context->_conf.set_val("key", client_key);
        if (r < 0) {
            std::cerr << "[trace] init: set_val(key) failed r=" << r << std::endl;
            return r;
        }
        context->_conf.apply_changes(nullptr);
        common_init_finish(context.get());
        Dispatcher::cct = context.get();
        std::cerr << "[trace] init: CephContext configured"
                  << " mon_host'" << mon_host << "'"
                  << " client_name" << client_name << "'" << std::endl;

        std::cerr << "[trace] init: starting io_context_pool" << std::endl;
        pool = std::make_unique<ceph::async::io_context_pool>(1);
        pool->start(1);
        std::cerr << "[trace] init: creating MonClient" << std::endl;
        monc = std::make_unique<MonClient>(context.get(), *pool);
        r = monc->build_initial_monmap();
        if (r < 0) {
            std::cerr << "[trace] init: build_initial_monmap failed r=" << r << std::endl;
            return r;
        }
        std::cerr << "[trace] init: MonClient created, monmap built" << std::endl;
        std::cerr << "[trace] init: creating client messenger" << std::endl;
        messenger.reset(
            Messenger::create_client_messenger(context.get(), "kv-osdmap-pull"));
        if (!messenger) {
            std::cerr << "[trace] init: create_client_messenger returned null" << std::endl;
            return -ENOMEM;
        }
        monc->set_messenger(messenger.get());
        messenger->set_auth_client(monc.get());
        messenger->add_dispatcher_tail(this);
        messenger->start();
        messenger_started = true;
        monc->set_want_keys(CEPH_ENTITY_TYPE_MON | CEPH_ENTITY_TYPE_OSD);
        std::cerr << "[trace] init: messenger started, want_keys=MON|OSD" << std::endl;
        std::cerr << "[trace] init: monc->init()" << std::endl;
        r = monc->init();
        if (r < 0) {
            std::cerr << "[trace] init: monc->init() failed r=" << r << std::endl;
            return r;
        }
        monc_started = true;
        std::cerr << "[trace] init: monc->authenticate(20)" << std::endl;
        monc->authenticate(20);
        if (r < 0) {
            std::cerr << "[trace] init: monc->authenticate() failed r=" << r << std::endl;
            return r;
        }
        std::cerr << "[trace] init: monc authenticated, waiting for rotating keys"
                  << std::endl;
        // Ensure OSD service tickets are obtained so we can build an
        // OSD anthorizer when connecting to the OSD Agent.
        r = monc->wait_auth_rotating(20);
        if (r < 0) {
            std::cerr << "[trace] init: wait_auth_rotating() failed r=" << r << std::endl;
            return r;
        }
        std::cerr << "[trace] init: rotating keys obtained - init done" << std::endl;
        return 0;
    }

    int pull_osd_map(const std::string& target_addr_str,
                     epoch_t current_epoch,
                     unsigned wait_seconds)
    {
        entity_addrvec_t target_addrs;
        entity_addr_t addr;
        if (!addr.parse(target_addr_str) || !addr.is_ip()) {
            AGENT_LOG(logger, agent::LogLevel::ERROR,
                      "failed to parse target address: " + target_addr_str);
            return -EINVAL;
        }
        addr.set_type(entity_addr_t::TYPE_MSGR2);
        target_addrs.v = (addr);

        std::cerr << "[trace] pull: ENTER - target=" << target_addr_str
                  << " current_epoch=" << current_epoch
                  << " wait_seconds=" << wait_seconds << std::endl;
        // Peer type OSD: the OSD Agent's messenger entity is OSD, so we need
        // an OSD authorizer.  The OSD Agent sets stateless_server policy for
        // CEPH_ENTITY_TYPE_CLIENT peers (our entity type), which is what the
        // server sees during the accept handshake.
        std::cerr << "[trace] pull: calling connect_to(OSD, ...)" << std::endl;
        auto conn = messenger->connect_to(CEPH_ENTITY_TYPE_OSD, target_addrs);
        if (!conn) {
            std::cerr << "[trace] pull: connect_to returned null" << std::endl;
            AGENT_LOG(logger, LogLevel::ERROR,
                      "failed to create connection to OSD agent at " + target_addr_str);
            return -ECONNREFUSED;
        }
        std::cerr << "[trace] pull: connect_to returned conn=" << conn.get()
                  << " peer_addr=" << conn->get_peer_addr()
                  << " is_connected=" << conn->is_connected() << std::endl;

        // Give the async connection a moment to initiate, then send the
        // message - Ceph messenger queues it umtil the connection is ready.
        std::cerr << "[trace] pull: sleeping 500ms for async connect" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::cerr << "[trace] pull: after sleep - is_connected="
                  << conn->is_connected() << std::endl;

        // Build the pull request as an MOSDMap with empty maps/incremental_maps
        // and current_epoch encoded into the data bufferlist.  The OSD Agent
        // distinguishes pull requests from MON-pushed OSDMaps by checking
        // maps.empty() && incremental_maps.empty() && data_len > 0
        auto req = ceph::make_message<MOSDMap>();
        ceph::bufferlist::list req_data;
        ceph::encode(current_epoch, req_data);
        req->set_data(req_data);
        std::cerr << "[trace] pull: sending MOSDMap pull request type="
                  << req->get_type()
                  << " current_epoch=" << current_epoch
                  << " data_len=" << req_data.length() << std::endl;
        conn->send_message2(std::move(req));
        std::cerr << "[trace] pull: send_message2 returned"
                  << " is_connected=" << conn->is_connected()
                  << " - waiting for response (timeout=" << wait_seconds << "s)"
                  << std::endl;
        AGENT_LOG(logger, LogLevel::INFO,
                  "send OSDMap pull request (current_epoch=" +
                  std::to_string(current_epoch) +
                  ") to OSD agent at " + target_addr_str);

        {
            std::unique_lock guard(cv_lock);
            std::cerr << "[trace] pull: entering cv.wait_for" << std::endl;
            if (!cv.wait_for(guard,
                             std::chrono::seconds(wait_seconds),
                             [this] { return reply_received || error_flag; })) {
                std::cerr << "[trace] pull: cv.wait_for TIMED OUT" << std::endl;
                AGENT_LOG(logger, AGENT_LOG_ERR,
                         "timed out waiting for OSDMap response");
                return -ETIMEDOUT;
            }
            std::cerr << "[trace] pull: cv.wait_for woke up"
                      << " reply_received=" << reply_received
                      << " error_flag=" << error_flag << std::endl;
        }
        return error_flag ? -EIO : 0;
    }

    void print_view()
    {
        std::lock_guard guard(osdmap_lock);
        epoch_t epoch = osdmap.get_epoch();
        int max_osd = osdmap.get_max_osd();
        int count = 0;
        for (int osd = 0; osd <= max_osd; ++osd) {
            if (!osdmap.exists(osd)) {
                continue;
            }
            std::cout << "osd." << osd
                      << " up=" << (osdmap.is_up(osd) ? 1 : 0)
                      << " in=" << (osdmap.is_in(osd) ? 1 : 0);
            const auto& addrs = osdmap.get_addrs(osd);
            for (const auto& a : addrs.v) {
                std::cout << " addr=" << a.ip_only_to_str();
            }
            coputs << std::endl;
            ++count;
        }
        std::cout << "epoch=" << epoch << " osd_count" << count << std::endl;
    }

    epoch_t get_epoch()
    {
        std::lock_guard guard(osdmap_lock);
        return osdmap.get_epoch();
    }

    bool is_full_reply() const { return full_reply; }
    bool is_incremental_reply() const { return incremental_reply; }
    bool is_empty_reply() const { return empty_reply; }
    epoch_t get_reply_epoch() const { return reply_epoch; }

    // Returns 0 if the actual reply type matches the expected type,
    // non-zero otherwise.  "full", "incremental", "empty" are accepted.
    int check_expext(const std::string& expected) const
    {
        if (expected == "full" && is_full_reply()) return 0;
        if (expected == "incremental" && is_incremental_reply()) return 0;
        if (expected == "empty" && is_empty_reply()) return 0;
        return -1;
    }

    std::string reply_type() const
    {
        if (is_full_reply()) return "full";
        if (is_incremental_reply()) return "incremental";
        if (is_empty_reply()) return "empty";
        return "unknown";
    }

    bool ms_dispatch(Message *message) override
    {
        std::cerr << "[trace] ms_dispatch: ENTER type=" << message->get_type()
                  << " name=" << message->get_type_name()
                  << " src=" << message->get_source()
                  << " payload_len=" << message->get_payload().length()
                  << std::endl;
        if (message->get_type() == CEPH_MSG_OSD_MAP) {
            std::cerr << "[trace] ms_dispatch: unexpected type" << message->get_type()
                      << " name=" << message->get_type_name()
                      << " - dropping" << std::endl;
            message->put();
            return false;
        }
        auto *reply static_cast<MOSDMap *>(message);
        {
            std::lock_guard guard(osdmap_lock);

            std::cerr << "[trace] ms_dispatch: OSDMap reply"
                      << " maps=" << reply->maps.size()
                      << " incremental_maps=" << reply->incremental_maps.size()
                      << " newest_map=" << reply->newest_map
                      << " trim_lower_bound="
                      << reply->cluster_osdmap_trim_lower_bound
                      << std::endl;

            if (!reply->maps.empty()) {
                full_reply = true;
                for (auto& [e, bl] : reply->maps) {
                    std::cerr << "[trace] ms_dispatch: decoding full map epoch=" << e
                              << " blen=" << bl.length() << std::endl;
                    osdmap.decode(bl);
                    reply_epoch = e;
                }
                AGENT_LOG(logger agent::LogLevel::INFO,
                          "received full OSD map epoch " +
                          std::to_string(reply_epoch));
            } else if (!reply->incremental_maps.empty()) {
                incremental_reply = true;
                for (auto& [e, bl] : reply->incremental_maps) {
                    std::cerr << "[trace] ms_dispatch: applying incremental epoch=" << e
                              << " blen=" << bl.length()
                              << " current_epoch=" << osdmap.get_epoch() << std::endl;
                    if (osdmap.get_epoch() == e - 1) {
                        OSDMap::Incremental inc(bl);
                        osdmap.apply_incremental(inc);
                    }
                    reply_epoch = e;
                }
                AGENT_LOG(logger agent::LogLevel::INFO,
                          "received incremental OSD map, applied up to epoch " +
                          std::to_string(reply_epoch));
            } else {
                empty_reply = true;
                reply_epoch = reply->newest_map;
                AGENT_LOG(logger agent::LogLevel::INFO,
                          "already up todate at epoch " +
                          std::to_string(reply_epoch));
            }
        }
        {
            std::lock_guard guard(cv_lock);
            reply_received = true;
        }
        cv.notify_all();
        std::cerr << "[trace] ms_dispatch: EXIT - notifying waiter" << std::endl;
        message->put();
        return true;
    }

    bool ms_handle_reset(Connection *c) override {
        std::cerr << "[trace] ms_handle_reset: peer=" << c->get_peer_addr()
                  << " - connection reset by peer" << std::endl;
        return false;
    }
    void ms_handle_remote_reset(Connection *c) override {
        std::cerr << "[trace] ms_handle_remote_reset: peer=" << c->get_peer_addr()
                  << std::endl;
    }
    bool ms_handle_refused(Connection *c) override {
        std::cerr << "[trace] ms_handle_refused: connection refused by peer"
                  << " peer=" << c->get_peer_addr() << std::endl;
        std::lock_guard guard(cv_lock);
        error_flag = true;
        cv.notify_all();
        return false;
    }
    void ms_handle_connect(Connection *c) override {
        std::cerr << "[trace] ms_handle_connect: peer=" << c->get_peer_addr()
                  << " is_connected=" << c->is_connected() << std::endl;
    }
    void ms_handle_fast_connect(Connection *c) override {
        std::cerr << "[trace] ms_handle_fast_connect: peer=" << c->get_peer_addr()
                  << std::endl;
    }

private:
    void shutdown()
    {
        {
            std::lock_guard guard(cv_lock);
            error_flag = true;
        }
        cv.notify_all();
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
    }

    agent::AgentLogger *logger = nullptr;
    boost::intrusive_ptr<CephContext> context;
    std::unique_ptr<ceph::async::io_context_pool> pool;
    std::unique_ptr<MonClient> monc;
    std::unique_ptr<Messenger> messenger;

    mutable std::mutex osdmap_lock;
    OSDMap osdmap;

    std::mutex cv_lock;
    std::condition_variable cv;
    bool reply_received = false;
    bool error_flag = false;

    bool full_reply = false;
    bool incremental_reply = false;
    bool empty_reply = false;
    epoch_t reply_epoch = 0;

    bool messenger_started = false;
    bool monc_started = false;
};

const char* get_arg(int argc, const char** argv, const std::string& name)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return nullptr;
}

} // namespace

int main(int argc, const char** argv)
{
    const char* mon_host = get_arg(argc, argv, "--mon-host");
    const char *client_name = get_arg(argc, argv, "--client-name");
    const char *client_key = get_arg(argc, argv, "--client-key");
    const char *target_addr = get_arg(argc, argv, "--target-addr");
    const char *current_epoch_str = get_arg(argc, argv, "--current-epoch");
    const char *log_dir = get_arg(argc, argv, "--log-dir");
    const char *wait_arg = get_arg(argc, argv, "--wait");
    const char *expect_arg = get_arg(argc, argv, "--expect");

    if (!mon_host || !client_name || !client_key || !target_addr ||
        !current_epoch_str) {
        std::cerr << "usage: kv-osd-pull-smoke --mon-host <host> "
                     "--client-name <client.name> --client-key <cephx-key> "
                     "--target-addr <ip[:port]> --current-epoch <N> "
                     "[--log-dir <directory>] [--wait <seconds>] "
                     "[--expect full|incremental|empty]"
                  << std::endl;
        return EXIT_FAILURE;
    }

    unsigned long current_epoch_val =
        std::strtoul(current_epoch_str, nullptr, 10);
    if (current_epoch_val > UINT32_MAX) {
        std::cerr << "Invalid current-epoch: " << current_epoch_str << std::endl;
        return EXIT_FAILURE;
    }
    unsigned wait_seconds = wait_arg ? std::strtoul(wait_arg, nullptr, 10) : 10;
    if (wait_seconds == 0) {
        wait_seconds = 10;
    }

    std::cerr << "[trace] main: args parsed"
              << " mon_host='" << mon_host << "'"
              << " client_name='" << client_name << "'"
              << " target_addr='" << target_addr << "'"
              << " current_epoch=" << current_epoch_val
              << " wait=" << wait_seconds << "s"
              << " log_dir='" << (log_dir ? log_dir : "(null)") << "'"
              << std::endl;

    agent::AgentLogger logger;
    int log_r = logger.init(log_dir, "kv_osdmap_pull");
    if (log_r < 0) {
        std::cerr << "[trace] main: logger init failed: " << log_r << std::endl;
        return EXIT_FAILURE;
    }
    std::cerr << "[trace] main: logger initialized" << std::endl;

    OSDPullAgent pull_agent;
    std::cerr << "[trace] main: calling pull_agent.init()" << std::endl;
    int r = pull_agent.init(mon_host, client_name, client_key, &logger);
    if (r < 0) {
        std::cerr << "[trace] main: init failed: " << cpp_strerror(r) << std::endl;
        return EXIT_FAILURE;
    }
    std::cerr << "[trace] main: init succeeded" << std::endl;

    std::cerr << "[trace] main: calling pull_agent.pull_osdmap()" << std::endl;
    r = pull_agent.pull_osdmap(
        target_addr,
        static_cast<epoch_t>(current_epoch_val),
        wait_seconds);
    if (r < 0) {
        std::cerr << "[trace] main: pull failed: " << cpp_strerror(r) << std::endl;
        return EXIT_FAILURE;
    }
    std::cerr << "[trace] main: pull succeeded, printing view" << std::endl;

    pull_agent.print_view();

    if (pull_agent.is_full_reply()) {
        std::cout << "result: full OSD map epoch "
                  << pull_agent.get_reply_epoch() << std::endl;
    } else if (pull_agent.is_incremental_reply()) {
        std::cout << "result: incremental OSD map epoch "
                  << pull_agent.get_reply_epoch() << std::endl;
    } else if (pull_agent.is_empty_reply()) {
        std::cout << "result: already up to date at epoch "
                  << pull_agent.get_reply_epoch() << std::endl;
    } else {
        std::cout << "result: unknow " << std::endl;
    }

    // If --expect is specified, verify the reply type matches and return
    // a non-zero exit code on mismatch so scripts can automate pass/fail.
    if (expect_arg) {
        std::string expected(expect_arg);
        if (pull_agent.check_expext(expected) != 0) {
            std::cerr << "FAIL: expected '" << expected
                      << "' but got '" << pull_agent.reply_type() << "'"
                      << std::endl;
            return EXIT_FAILURE;
        }
        std::cerr << "PASS: expected '" << expected << "' matched" << std::endl;
    }

    return EXIT_SUCCESS;
}
