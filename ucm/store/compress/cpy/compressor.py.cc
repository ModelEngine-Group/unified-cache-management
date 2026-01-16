#include "compressor.h"
#include "template/store_binder.h"

PYBIND11_MODULE(ucmcompressor, module)
{
    namespace py = pybind11;
    using namespace UC::Compressor;
    using CompressorPy = UC::Detail::StoreBinder<Compressor, Config>;
    module.attr("project") = UCM_PROJECT_NAME;
    module.attr("version") = UCM_PROJECT_VERSION;
    module.attr("commit_id") = UCM_COMMIT_ID;
    module.attr("build_type") = UCM_BUILD_TYPE;
    auto store = py::class_<CompressorPy, std::unique_ptr<CompressorPy>>(module, "Compressor");
    auto config = py::class_<Config>(store, "Config");
    config.def(py::init<>());
    config.def_readwrite("storeBackend", &Config::storeBackend);
    config.def_readwrite("uniqueId", &Config::uniqueId);
    config.def_readwrite("deviceId", &Config::deviceId);
    config.def_readwrite("tensorSize", &Config::tensorSize);
    config.def_readwrite("shardSize", &Config::shardSize);
    config.def_readwrite("blockSize", &Config::blockSize);
    config.def_readwrite("streamNumber", &Config::streamNumber);
    // config.def_readwrite("bufferSize", &Config::bufferSize);
    // config.def_readwrite("shareBufferEnable", &Config::shareBufferEnable);
    // config.def_readwrite("waitingQueueDepth", &Config::waitingQueueDepth);
    // config.def_readwrite("runningQueueDepth", &Config::runningQueueDepth);
    config.def_readwrite("timeoutMs", &Config::timeoutMs);
    store.def(py::init<>());
    store.def("Self", &CompressorPy::Self);
    store.def("Setup", &CompressorPy::Setup);
    store.def("Lookup", &CompressorPy::Lookup, py::arg("ids").noconvert());
    store.def("Prefetch", &CompressorPy::Prefetch, py::arg("ids").noconvert());
    store.def("Load", &CompressorPy::Load, py::arg("ids").noconvert(),
              py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    store.def("Dump", &CompressorPy::Dump, py::arg("ids").noconvert(),
              py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    store.def("Check", &CompressorPy::Check);
    store.def("Wait", &CompressorPy::Wait);
}
