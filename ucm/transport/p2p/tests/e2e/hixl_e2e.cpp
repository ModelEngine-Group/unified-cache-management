#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "core/transport_manager.h"
#include "test_common.h"

using namespace transport;

namespace {

constexpr size_t kLen = 4 * 1024;

const char* envText(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr || *value == '\0' ? fallback : value;
}

int envInt(const char* name, int fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') { return fallback; }
    char* end = nullptr;
    const auto value = std::strtol(text, &end, 10);
    return end != nullptr && *end == '\0' ? static_cast<int>(value) : fallback;
}

int32_t envHixlPort(const char* name, int32_t fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') { return fallback; }
    char* end = nullptr;
    const auto value = std::strtol(text, &end, 10);
    if (end == nullptr || *end != '\0' || value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
        return fallback;
    }
    return static_cast<int32_t>(value);
}

std::vector<int> envIntList(const char* name, std::vector<int> fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') { return fallback; }

    std::vector<int> result;
    std::istringstream iss(text);
    std::string item;
    while (std::getline(iss, item, ',')) {
        if (item.empty()) { return fallback; }
        char* end = nullptr;
        const auto value = std::strtol(item.c_str(), &end, 10);
        if (end == nullptr || *end != '\0') { return fallback; }
        result.push_back(static_cast<int>(value));
    }
    return result.empty() ? fallback : result;
}

std::string joinDeviceIds(const std::vector<int>& device_ids)
{
    std::ostringstream oss;
    for (size_t i = 0; i < device_ids.size(); ++i) {
        if (i != 0) { oss << ','; }
        oss << device_ids[i];
    }
    return oss.str();
}

struct Config {
    std::string local_host =
        envText("HIXL_TEST_LOCAL_HOST", envText("HIXL_TEST_HOST", "127.0.0.1"));
    std::string peer_host = envText("HIXL_TEST_PEER_HOST", envText("HIXL_TEST_HOST", "127.0.0.1"));
    uint16_t server_manager_port = test::envPort("TRANSPORT_TEST_PORT_A", 4501);
    uint16_t client_manager_port = test::envPort("TRANSPORT_TEST_PORT_B", 4502);
    uint16_t server_control_port = test::envPort("TRANSPORT_CONTROL_PORT_A", 4601);
    uint16_t client_control_port = test::envPort("TRANSPORT_CONTROL_PORT_B", 4602);
    int32_t server_hixl_port = envHixlPort("HIXL_TEST_PORT_A", -1);
    int32_t client_hixl_port = envHixlPort("HIXL_TEST_PORT_B", -1);
    int server_device_id = envInt("HIXL_TEST_DEVICE_A", 4);
    int client_device_id = envInt("HIXL_TEST_DEVICE_B", 5);
    std::vector<int> client_device_ids =
        envIntList("HIXL_TEST_DEVICE_IDS_B",
                   envIntList("HIXL_TEST_DEVICES_B", std::vector<int>{client_device_id}));
    int connect_timeout_ms = envInt("HIXL_TEST_CONNECT_TIMEOUT_MS", 30000);
    int transfer_timeout_ms = envInt("HIXL_TEST_TRANSFER_TIMEOUT_MS", 30000);
    int wait_attempts = envInt("HIXL_TEST_WAIT_ATTEMPTS", 600);
    int wait_interval_ms = envInt("HIXL_TEST_WAIT_RETRY_MS", 100);
};

class AclRuntime {
public:
    explicit AclRuntime(int device_id) : device_id_(device_id)
    {
        std::cerr << "[HIXL e2e] aclInit begin\n";
        const auto init_status = aclInit(nullptr);
        std::cerr << "[HIXL e2e] aclInit return " << static_cast<int>(init_status) << "\n";
        if (init_status != ACL_ERROR_NONE) { return; }

        ok_ = SetDevice(device_id_);
    }

    ~AclRuntime()
    {
        if (ok_) {
            for (const auto device_id : device_ids_) { aclrtResetDevice(device_id); }
            aclFinalize();
        }
    }

    AclRuntime(const AclRuntime&) = delete;
    AclRuntime& operator=(const AclRuntime&) = delete;

    bool ok() const { return ok_; }

    bool SetDevice(int device_id)
    {
        std::cerr << "[HIXL e2e] aclrtSetDevice(" << device_id << ") begin\n";
        const auto set_device_status = aclrtSetDevice(device_id);
        std::cerr << "[HIXL e2e] aclrtSetDevice(" << device_id << ") return "
                  << static_cast<int>(set_device_status) << "\n";
        if (set_device_status != ACL_ERROR_NONE) { return false; }
        if (std::find(device_ids_.begin(), device_ids_.end(), device_id) == device_ids_.end()) {
            device_ids_.push_back(device_id);
        }
        return true;
    }

private:
    int device_id_ = 0;
    std::vector<int> device_ids_;
    bool ok_ = false;
};

ManagerID makeManagerID(const Config& config, bool server)
{
    return config.local_host + ":" +
           std::to_string(server ? config.server_manager_port : config.client_manager_port);
}

ManagerID peerManagerID(const Config& config, bool server)
{
    return config.peer_host + ":" +
           std::to_string(server ? config.client_manager_port : config.server_manager_port);
}

Endpoint peerEndpoint(const Config& config, bool server)
{
    return test::makeEndpoint(config.peer_host,
                              server ? config.client_control_port : config.server_control_port);
}

HixlInitAttrs makeHixlAttrs(const Config& config, bool server)
{
    HixlInitAttrs attrs;
    attrs.ip = config.local_host;
    const auto device_ids =
        server ? std::vector<int>{config.server_device_id} : config.client_device_ids;
    const auto base_port = server ? config.server_hixl_port : config.client_hixl_port;
    for (size_t i = 0; i < device_ids.size(); ++i) {
        HixlInitAttrs::Instance instance;
        const auto port_value = static_cast<int64_t>(base_port) + static_cast<int64_t>(i);
        instance.port = base_port < 0 || port_value < std::numeric_limits<int32_t>::min() ||
                                port_value > std::numeric_limits<int32_t>::max()
                            ? -1
                            : static_cast<int32_t>(port_value);
        instance.device_id = device_ids[i];
        attrs.instances.push_back(std::move(instance));
    }
    attrs.connect_timeout_ms = config.connect_timeout_ms;
    attrs.transfer_timeout_ms = config.transfer_timeout_ms;
    return attrs;
}

std::string describeHixlInstances(const HixlInitAttrs& attrs)
{
    std::ostringstream oss;
    for (size_t i = 0; i < attrs.instances.size(); ++i) {
        if (i != 0) { oss << ";"; }
        oss << attrs.ip << ':' << attrs.instances[i].port
            << "/device=" << attrs.instances[i].device_id;
    }
    return oss.str();
}

bool exchangeMetadataWithRetry(TransportManager& manager, const ManagerID& manager_id,
                               const Config& config)
{
    for (int attempt = 1; attempt <= config.wait_attempts; ++attempt) {
        if (manager.ExchangeMetadata(manager_id) == Status::OK()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(config.wait_interval_ms));
    }
    return false;
}

struct RemoteDeviceAddress {
    int device_id = -1;
    uint64_t address = 0;
};

struct LocalDeviceMemory {
    int device_id = -1;
    void* address = nullptr;
    MemoryHandle handle = kInvalidMemoryHandle;
};

bool parseAddressMessage(const std::string& message, std::vector<RemoteDeviceAddress>& addresses)
{
    std::istringstream iss(message);
    std::string tag;
    iss >> tag;
    addresses.clear();
    if (tag == "ADDR") {
        uint64_t address = 0;
        iss >> std::hex >> address;
        if (address == 0) { return false; }
        addresses.push_back(RemoteDeviceAddress{-1, address});
        return true;
    }

    if (tag != "ADDRS") { return false; }
    size_t count = 0;
    iss >> std::dec >> count;
    if (count == 0) { return false; }
    addresses.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        RemoteDeviceAddress item;
        iss >> std::dec >> item.device_id >> std::hex >> item.address;
        if (!iss || item.address == 0) { return false; }
        addresses.push_back(item);
    }
    return addresses.size() == count;
}

bool sameHost(const Config& config)
{
    return config.local_host == config.peer_host ||
           (config.local_host == "127.0.0.1" && config.peer_host == "localhost") ||
           (config.local_host == "localhost" && config.peer_host == "127.0.0.1");
}

bool validateDeviceConfig(const Config& config)
{
    if (config.client_device_ids.empty()) {
        std::cerr << "HIXL e2e invalid device config: client device list is empty\n";
        return false;
    }
    if (config.server_device_id < 0 ||
        std::any_of(
            config.client_device_ids.begin(), config.client_device_ids.end(),
            [](int device_id) { return device_id < 0; })) {
        std::cerr << "HIXL e2e invalid device config: device id must be non-negative\n";
        return false;
    }
    if (sameHost(config) &&
        std::find(config.client_device_ids.begin(), config.client_device_ids.end(),
                  config.server_device_id) != config.client_device_ids.end()) {
        std::cerr << "HIXL e2e invalid device config: same-host two-process test cannot use the "
                     "same device_id="
                  << config.server_device_id
                  << ". Set HIXL_TEST_DEVICE_A and HIXL_TEST_DEVICE_IDS_B to disjoint "
                     "devices.\n";
        return false;
    }
    return true;
}

bool verifyPattern(const unsigned char* data, const std::string& label)
{
    size_t first_error = kLen;
    for (size_t i = 0; i < kLen; ++i) {
        if (data[i] != static_cast<unsigned char>(i & 0xff)) {
            first_error = i;
            break;
        }
    }

    if (first_error == kLen) {
        std::cout << "[A] >>> VERIFY " << label << ": PASS <<<\n";
        return true;
    }

    std::cerr << "[A] >>> VERIFY " << label << ": FAIL <<< at index " << first_error
              << ", expect=" << int(first_error & 0xff) << ", actual=" << int(data[first_error])
              << "\n";
    return false;
}

bool executeAsync(TransportManager& manager, const Operation& batch,
                  const std::string& submit_label, const std::string& wait_label)
{
    TransferHandle handle = kInvalidTransferHandle;
    if (!test::expectOk(manager.ExecuteAsync(batch, handle), submit_label.c_str())) {
        return false;
    }

    for (;;) {
        TransferStatus status = TransferStatus::Failed;
        if (!test::expectOk(manager.GetStatus(handle, status), wait_label.c_str())) {
            return false;
        }
        if (status == TransferStatus::Completed) { return true; }
        if (status == TransferStatus::Failed) {
            std::cerr << wait_label << " failed: transfer failed\n";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int runServer()
{
    Config config;
    if (!validateDeviceConfig(config)) { return 1; }
    std::cerr << "[HIXL e2e server] init standalone TCP listen " << config.local_host << ':'
              << config.server_control_port << '\n';
    TcpMessageChannel control;
    if (!test::expectOk(
            control.Init(test::makeEndpoint(config.local_host, config.server_control_port)),
            "server init standalone TCP")) {
        return 1;
    }

    std::cerr << "[HIXL e2e server] init ACL device " << config.server_device_id << '\n';
    AclRuntime acl(config.server_device_id);
    if (!test::expectTrue(acl.ok(), "server initialize ACL runtime")) { return 1; }

    const auto manager_id = makeManagerID(config, true);
    const auto hixl_attrs = makeHixlAttrs(config, true);
    std::cerr << "[HIXL e2e server] manager_id=" << manager_id
              << " peer_control=" << config.peer_host << ':' << config.client_control_port
              << " hixl_instance=\"" << describeHixlInstances(hixl_attrs) << "\"\n";

    TransportManager manager(manager_id);
    if (!test::expectOk(manager.Init(), "server init manager") ||
        !test::expectOk(manager.InstallTransport(TransportProtocol::Hixl, hixl_attrs),
                        "server install HIXL")) {
        return 1;
    }

    std::cerr << "[HIXL e2e server] allocate/register device memory\n";
    LocalDeviceMemory device_memory;
    device_memory.device_id = config.server_device_id;
    if (!test::expectTrue(
            aclrtMalloc(&device_memory.address, kLen, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_ERROR_NONE,
            "server allocate device memory") ||
        !test::expectTrue(aclrtMemset(device_memory.address, kLen, 0, kLen) == ACL_ERROR_NONE,
                          "server clear device memory")) {
        return 1;
    }

    MemoryRegion device_region;
    device_region.addr = device_memory.address;
    device_region.length = kLen;
    device_region.type = MemoryType::Device;
    device_region.device_id = config.server_device_id;
    if (!test::expectOk(manager.RegisterMemory(device_region, device_memory.handle),
                        "server register device memory")) {
        return 1;
    }

    std::cout << "[B] waiting READY from client on standalone TCP port "
              << config.server_control_port << "\n";
    Endpoint client_control;
    std::string ready;
    if (!test::receiveText(control, client_control, ready,
                           "server receives client READY over standalone TCP") ||
        ready != "READY") {
        return 1;
    }
    std::cout << "[B] client READY from " << client_control.host << ':' << client_control.port
              << "\n";

    std::ostringstream address_message;
    address_message << "ADDRS 1 " << std::dec << config.server_device_id << " " << std::hex
                    << reinterpret_cast<uintptr_t>(device_memory.address);
    if (!test::sendTextWithRetry(control, client_control, address_message.str(),
                                 config.wait_attempts, config.wait_interval_ms,
                                 "server sends device addresses over standalone TCP")) {
        return 1;
    }

    std::cout << "[B] waiting DONE from client\n";
    std::string done;
    if (!test::receiveText(control, done, "server receives completion over standalone TCP") ||
        done != "DONE") {
        return 1;
    }

    unsigned char back[16] = {};
    if (!test::expectTrue(aclrtMemcpy(back, sizeof(back), device_memory.address, sizeof(back),
                                      ACL_MEMCPY_DEVICE_TO_HOST) == ACL_ERROR_NONE,
                          "server copy final device bytes to host")) {
        return 1;
    }
    std::cout << "[B] first bytes after A->B: ";
    for (int i = 0; i < 16; ++i) { std::cout << int(back[i]) << " "; }
    std::cout << "\n";

    if (!test::expectOk(manager.Shutdown(), "server shutdown manager") ||
        !test::expectOk(control.Shutdown(), "server shutdown standalone TCP")) {
        return 1;
    }
    aclrtFree(device_memory.address);
    return 0;
}

int runClient()
{
    Config config;
    if (!validateDeviceConfig(config)) { return 1; }
    std::cerr << "[HIXL e2e client] init standalone TCP listen " << config.local_host << ':'
              << config.client_control_port << '\n';
    TcpMessageChannel control;
    if (!test::expectOk(
            control.Init(test::makeEndpoint(config.local_host, config.client_control_port)),
            "client init standalone TCP")) {
        return 1;
    }

    if (!test::sendTextWithRetry(control, peerEndpoint(config, false), "READY",
                                 config.wait_attempts, config.wait_interval_ms,
                                 "client sends READY over standalone TCP")) {
        return 1;
    }

    std::cerr << "[HIXL e2e client] init ACL device " << config.client_device_ids.front()
              << " device_ids=" << joinDeviceIds(config.client_device_ids) << '\n';
    AclRuntime acl(config.client_device_ids.front());
    if (!test::expectTrue(acl.ok(), "client initialize ACL runtime")) { return 1; }

    const auto manager_id = makeManagerID(config, false);
    const auto hixl_attrs = makeHixlAttrs(config, false);
    std::cerr << "[HIXL e2e client] manager_id=" << manager_id
              << " peer_control=" << config.peer_host << ':' << config.server_control_port
              << " hixl_instances=\"" << describeHixlInstances(hixl_attrs) << "\"\n";

    TransportManager manager(manager_id);
    if (!test::expectOk(manager.Init(), "client init manager") ||
        !test::expectOk(manager.InstallTransport(TransportProtocol::Hixl, hixl_attrs),
                        "client install HIXL")) {
        return 1;
    }

    std::cerr << "[HIXL e2e client] allocate/register host memory\n";
    void* host = nullptr;
    if (!test::expectTrue(aclrtMallocHost(&host, kLen) == ACL_ERROR_NONE,
                          "client allocate host memory")) {
        return 1;
    }
    auto* p = static_cast<unsigned char*>(host);
    for (size_t i = 0; i < kLen; ++i) { p[i] = static_cast<unsigned char>(i & 0xff); }

    MemoryRegion host_region;
    host_region.addr = host;
    host_region.length = kLen;
    host_region.type = MemoryType::Host;
    MemoryHandle host_handle = kInvalidMemoryHandle;
    if (!test::expectOk(manager.RegisterMemory(host_region, host_handle),
                        "client register host memory")) {
        return 1;
    }

    std::cout << "[A] waiting server device address on standalone TCP port "
              << config.client_control_port << "\n";
    std::string address_message;
    std::vector<RemoteDeviceAddress> remote_devices;
    if (!test::receiveText(control, address_message,
                           "client receives device addresses over standalone TCP") ||
        !test::expectTrue(parseAddressMessage(address_message, remote_devices),
                          "client parses server device addresses")) {
        return 1;
    }
    for (const auto& remote_device : remote_devices) {
        std::cout << "[A] remote B device " << remote_device.device_id << " addr = 0x" << std::hex
                  << remote_device.address << std::dec << "\n";
    }

    const auto peer_manager_id = peerManagerID(config, false);
    if (!test::expectTrue(exchangeMetadataWithRetry(manager, peer_manager_id, config),
                          "client exchanges HIXL metadata with server")) {
        return 1;
    }
    if (!test::expectOk(manager.Connect(TransportProtocol::Hixl, peer_manager_id),
                        "client connects HIXL peer")) {
        return 1;
    }

    bool verify_ok = true;
    for (const auto& remote_device : remote_devices) {
        Operation batch;
        batch.target_manager = peer_manager_id;
        batch.direct = OperationDirect::RemoteDeviceHost;
        batch.opcode = Opcode::Write;
        batch.ops.push_back(Segment{host_region.addr, remote_device.address, kLen});

        for (size_t i = 0; i < kLen; ++i) { p[i] = static_cast<unsigned char>(i & 0xff); }
        std::cout << "[A] Host -> B Device " << remote_device.device_id << " WRITE sync\n";
        if (!test::expectOk(manager.ExecuteSync(batch),
                            "client manager routes HIXL host to remote device write sync")) {
            return 1;
        }
        std::cout << "[A] H2D WRITE sync done device=" << remote_device.device_id << "\n";

        std::memset(host, 0, kLen);
        batch.opcode = Opcode::Read;
        std::cout << "[A] B Device " << remote_device.device_id << " -> Host READ sync\n";
        if (!test::expectOk(manager.ExecuteSync(batch),
                            "client manager routes HIXL remote device to host read sync")) {
            return 1;
        }
        std::cout << "[A] D2H READ sync done device=" << remote_device.device_id << "\n";
        verify_ok =
            verifyPattern(p, "sync device " + std::to_string(remote_device.device_id)) && verify_ok;

        for (size_t i = 0; i < kLen; ++i) { p[i] = static_cast<unsigned char>(i & 0xff); }
        batch.opcode = Opcode::Write;
        std::cout << "[A] Host -> B Device " << remote_device.device_id << " WRITE async\n";
        if (!executeAsync(manager, batch,
                          "client manager submits HIXL host to remote device write async",
                          "client manager waits HIXL host to remote device write async")) {
            return 1;
        }
        std::cout << "[A] H2D WRITE async done device=" << remote_device.device_id << "\n";

        std::memset(host, 0, kLen);
        batch.opcode = Opcode::Read;
        std::cout << "[A] B Device " << remote_device.device_id << " -> Host READ async\n";
        if (!executeAsync(manager, batch,
                          "client manager submits HIXL remote device to host read async",
                          "client manager waits HIXL remote device to host read async")) {
            return 1;
        }
        std::cout << "[A] D2H READ async done device=" << remote_device.device_id << "\n";

        verify_ok = verifyPattern(p, "async device " + std::to_string(remote_device.device_id)) &&
                    verify_ok;
    }

    if (!test::sendTextWithRetry(control, peerEndpoint(config, false), "DONE", config.wait_attempts,
                                 config.wait_interval_ms,
                                 "client sends completion over standalone TCP")) {
        return 1;
    }

    if (!test::expectOk(manager.Shutdown(), "client shutdown manager") ||
        !test::expectOk(control.Shutdown(), "client shutdown standalone TCP")) {
        return 1;
    }
    aclrtFreeHost(host);
    return verify_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : envText("HIXL_TEST_ROLE", "");
    if (mode == "server" || mode == "B") { return runServer(); }
    if (mode == "client" || mode == "A") { return runClient(); }

    std::cerr << "usage: " << argv[0] << " server|client\n";
    return 1;
}
