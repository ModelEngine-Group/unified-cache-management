#include <torch/extension.h>
#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hamming_score", &(kvlib::HammingScoreContiCUDA), "some comments");
}