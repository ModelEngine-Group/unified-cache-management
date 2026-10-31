/*
 * Precompiled header for the gsa_prefetch test suite.
 *
 * Pulls in the heavy torch / pybind11 / ATen headers once into a .gch so each
 * test / production source compiles in seconds rather than minutes.  The
 * include guards in these headers make the subsequent `#include` in the real
 * sources a no-op after the PCH is loaded.
 */
#ifndef UCM_PREFETCH_TEST_PCH_H
#define UCM_PREFETCH_TEST_PCH_H

#include <ATen/Tensor.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

#endif  // UCM_PREFETCH_TEST_PCH_H
