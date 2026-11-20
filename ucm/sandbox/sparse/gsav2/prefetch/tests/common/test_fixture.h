/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * Shared helpers for the prefetch test suite.
 */
#ifndef UCM_PREFETCH_TEST_FIXTURE_H
#define UCM_PREFETCH_TEST_FIXTURE_H

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "kvcache_pre.h"
#include "kvcache_trans_ascend.h"
#include "kvcache_trans_ascend_stub.h"
#include "mock_store_v1.h"

namespace ucmprefetch_test {

/* Build a std::vector<int> of the 11 modelInfo slots the engine expects. */
inline std::vector<int> MakeModelInfo(int layerNum = 2, int maxBlockLen = 4096, int spaTopkLen = 64,
                                      int streamNum = 1, int sparseLen = 128, int prefetchType = 1,
                                      int dpRank = 99001, int tpSize = 1, int indexerOffload = 0,
                                      int isSfaC8 = 0, int maxBs = 4)
{
    return {layerNum, maxBlockLen, spaTopkLen,     streamNum, sparseLen, prefetchType,
            dpRank,   tpSize,      indexerOffload, isSfaC8,   maxBs};
}

/* Build a std::vector<int> of the 5 sizeinfo slots. */
inline std::vector<int> MakeSizeInfo(int blockSize = 4, int numHead = 1, int kDim = 4, int vDim = 4,
                                     int elementSize = 2)
{ return {blockSize, numHead, kDim, vDim, elementSize}; }

/* Unique shared-memory names per test avoid cross-test collisions when the
 * previous run did not clean up. Returns a never-used dp rank id. */
inline int UniqueDpRank()
{
    static int counter = 90000;
    return __sync_add_and_fetch(&counter, 1);
}

/* Pre-remove any leftover shared memory object for the engine. */
inline void CleanupShmFor(int dpRank)
{
    std::string name = "gsa_prefetch_result_dp" + std::to_string(dpRank);
    shm_unlink(name.c_str());
    std::string packName = "gsa_kv_pack_control_dp" + std::to_string(dpRank);
    shm_unlink(packName.c_str());
}

/* A writeable temp log path that is removed when the guard goes out of scope. */
class TempLogFile {
public:
    explicit TempLogFile(std::string tag = "prefetch_ut")
    {
        char tmpl[] = "/tmp/opencode_prefetch_XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd >= 0) {
            ::close(fd);
            path_ = tmpl;
        }
        std::ofstream touch(path_, std::ios::app);
    }
    ~TempLogFile()
    {
        if (!path_.empty()) { std::remove(path_.c_str()); }
    }
    const std::string& Path() const { return path_; }

private:
    std::string path_;
};

}  // namespace ucmprefetch_test

#endif  // UCM_PREFETCH_TEST_FIXTURE_H
