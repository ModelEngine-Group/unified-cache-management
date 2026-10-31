/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * Test-only stub for the Huawei Secure C header.  The prefetch sources
 * `#include "securec.h"` but never call any securec symbol, so an empty
 * header lets the test build proceed without libsecurec installed.  If secure
 * variants are ever required, define them here as thin wrappers over the
 * standard C library.
 */
#ifndef UCM_PREFETCH_TEST_SECUREC_STUB_H
#define UCM_PREFETCH_TEST_SECUREC_STUB_H

#include <cstdarg>
#include <cstdio>
#include <cstring>

#endif  // UCM_PREFETCH_TEST_SECUREC_STUB_H
