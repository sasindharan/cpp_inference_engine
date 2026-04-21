#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "core/inference.h"

namespace py = pybind11;

py::array_t<float> run_model_numpy(py::array_t<float> input_array) {
    auto buf = input_array.request();

    float* ptr = (float*)buf.ptr;
    std::vector<float> input(ptr, ptr + buf.size);

    auto output = run_inference(input);

    return py::array_t<float>(output.size(), output.data());
}

PYBIND11_MODULE(cpp_engine, m) {
    m.def("run_model", &run_model_numpy);
}
