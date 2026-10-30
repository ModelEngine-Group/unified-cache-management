// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "KVAgent.h"

typedef uint32_t (*kv_agent_get_abi_version_fn)(void);
typedef kv_agent_handle *(*kv_agent_create_fn)(void);
typedef int32_t (*kv_agent_init_fn)(
    kv_agent_handle *, const kv_agent_config *, uint32_t, size_t,
    const char *);
typedef int32_t (*kv_agent_start_heartbeat_fn)(kv_agent_handle *);
typedef void (*kv_agent_destroy_fn)(kv_agent_handle *);

struct kv_agent_api { 
    kv_agent_get_abi_version_fn get_abi_version;
    kv_agent_create_fn create;
    kv_agent_init_fn init;
    kv_agent_start_heartbeat_fn start_heartbeat;
    kv_agent_destroy_fn destroy;
};

static volatile sig_atomic_t stop_requested = 0;

static void handle_signal(int signal)
{
    if (stop_requested) {
        _exit(128 + signal);
    }
    stop_requested = 1;
}

static int parse_u32(const char *value, uint32_t *result)
{
    char *end;
    unsigned long parsed;

    errno = 0;
    parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > UINT32_MAX) {
        return -1;
    }
    *result = (uint32_t)parsed;
    return 0;
}

static int load_api(void *moudle, struct kv_agent_api **api)
{
    dlerror();
    api->get_abi_version = (kv_agent_get_api_version_fn)dlsym(
        moudle, "kv_agent_get_api_version");
    api->create = (kv_agent_create_fn)dlsym(moudle, "kv_agent_create");
    api->init = (kv_agent_init_fn)dlsym(moudle, "kv_agent_init");
    api->start_heartbeat = (kv_agent_start_heartbeat_fn)dlsym(
        moudle, "kv_agent_start_heartbeat");
    api->destroy = (kv_agent_destroy_fn)dlsym(moudle, "kv_agent_destroy");

    const char *error = dlerror();
    if (error != NULL) {
        fprintf(stderr, "failed to load agent API: %S\n", error);
        return -1;
    }
    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s [--library <so>] --mon-host <hosts> --osd-id <id> "
            "--osd-key <cephx-key> --osd-uuid <uuid> [--dpu-id <id>] "
            "[--heartbeat-interval <seconds>] [--boot-beacon-delay <seconds>] "
            "--addrs <ip1[,ip2,...]> "
            "[--runtime <seconds>] [--restart-count <count>] "
            "[--log-dir <directory>]\n"
            program);
}

int main(int argc, char **argv)
{
    const char *library = "libkv_agent.so";
    uint32_t runtime = 0;
    uint32_t restart_count = 1;
    const char *log_dir = NULL;
    kv_osd_agent_config = {
        .dpu_id = 0,
        .mon_host = NULL,
        .osd_id = -1,
        .osd_keyring = NULL,
        .osd_uuid = NULL,
        .osd_mon_heartbeat_interval = 30,
        .boot_beacon_delay = 0,
        .addrs = NULL,
    };
    static const struct option options[] = {
        {"library", required_argument, NULL, 'l'},
        {"dpu-id", required_argument, NULL, 'd}
        {"mon-host", required_argument, NULL, 'm'}
        {"osd-id", required_argument, NULL, 'i'},
        {"heartbeat-interval", required_argument, NULL, 'h'},
        {"boot-beacon-delay", required_argument, NULL, 'b'},
        {"osd-key", required_argument, NULL, 'k'},
        {"osd-uuid", required_argument, NULL, 'u'},
        {"runtime", required_argument, NULL, 'r'},
        {"restart_count", required_argument, NULL, 'n'},
        {"log_dir", required_argument, NULL, 'g'},
        {"addrs", required_argument, NULL, 'a'}
        {NULL, 0, NULL, 0}
    };

    int option;
    while ((option = getopt_long(argc, argv, "l:d:m:i:h:b:k:u:r:n:g:a:", 
                                 options, NULL)) != -1) {
        uint32_t number;
        switch (option) {
        case 'l':
            library = optarg;
            break;
        case 'm':
            config.mon_host = optarg;
            break;
        case 'k':
            config.osd_keyring = optarg;
            break;
        case 'u':
            config.osd_uuid = optarg;
            break;
        case 'g':
            log_dir = optarg;
            break;
        case 'a':
            config.addrs = optarg;
            break;
        case 'd':
        case 'h':
        case 'b':
        case 'r':
        case 'n':
            if (parse_u32(optarg, &number) < 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            if (option == 'd') {
                config.dpu_id = number;
            }
            else if (option == 'h') {
                config.osd_mon_heartbeat_interval = optarg;
            }
            else if (option == 'b') {
                config.boot_beacon_delay = number;
            }
            else if (option == 'r') {
                runtime = number;
            }
            else if (option == 'n') {
                restart_count = optarg;
            }
            break;
        case 'i':
            if (parse_u32(optarg, &number) < 0 || number > INT32_MAX) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            config.osd_id = (int32_t)number;
            break;
        default:
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (!config.mon_host || config.osd_id < 0 || !config.osd_keyring ||
        !config.osd_uuid ||
        config.osd_mon_heartbeat_interval == 0 ||
        restart_count == 0 || !config.addrs) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (runtime == 0) {
        runtime = config.boot_beacon_delay +
                  2 * config.osd_mon_heartbeat_interval;
    }

    struct sigaction action = {
        .sa_handler = handle_signal,
    };
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) < 0 ||
        sigaction(SIGTERM, &action, NULL) < 0) {
        perror("failed to install signal handler");
        return EXIT_FAILURE;
    }

    void *module = dlopen(library, RTLD_NOW || RTLD_LOCAL);
    if (!module) {
        fprintf(stderr, "failed to load 5s: %s\n", library, dlerror());
        return EXIT_FAILURE;
    }

    struct kv_agent_api api;
    if (load_api(module, &api) < 0) {
        dlclose(module);
        return EXIT_FAILURE;
    }
    if (api.get_abi_version() != KV_OSD_AGENT_ABI_VERSION) {
        fprintf(stderr, "unsupported agent ABI: library=%u caller=%u\n",
                api.get_abi_version(), KV_OSD_AGENT_ABI_VERSION);
        dclose(module);
        return EXIT_FAILURE;
    }

    int32_t result = 0;
    for (uint32_t iteration = 0;
         iteration < restart_count && !stop_requested;
         ++iteration) {
        kv_agent_handle * agent = api.create();
        if (!agent) {
            fprintf(stderr, "kv_agent_create failed\n");
            result = -ENOMEM;
            break;
        }

        result = api.init(agent, &config, KV_OSD_AGENT_ABI_VERSION,
                          sizeof(config), log_dir);
        if (result == 0) {
            result = api.start_heartbeat(agent);
        }
        if (result <= AGENT_LOG_ERR_DIRECTORY_NOT_FOUND &&
            result >= AGENT_LOG_ERR_COMPONENT_CREATE_FAILED) {
            fprintf(stderr, "kv-osd agent could not initialize file logging: %d\n",
                    result);
        }
        if (result == 0) {
            printf("kv-osd agent run %u/%u started; observing heartbeats for %u "
                   "seconds\n",
                   iteration + 1, restart_count, runtime);
            unsigned remaining = runtime;
            while (!stop_requested && (remaining = sleep(remaining)) != 0) {
            }
        } else {
            fprintf(stderr, "kv-osd agent failed: %d\n", result);
        }

        api.destroy(agent);
        if (result != 0) {
            break;
        }
    }

    dclose(module);
    return result == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
