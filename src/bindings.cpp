#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "inference.h"

namespace py = pybind11;

PYBIND11_MODULE(llm_cpp_engine, m) {
    m.doc() = "XingLing小模型的C++推理引擎";

    py::class_<InferenceEngine>(m, "InferenceEngine")
        .def(py::init<>())
        .def("load", &InferenceEngine::load, "从目录下加载模型")
        .def("next_token", [](InferenceEngine& self, const std::vector<int>& input_ids, 
                              float temperature, float top_p, float repetition_penalty) {
            return self.next_token(input_ids, temperature, top_p, repetition_penalty);
        }, 
        py::arg("input_ids"), 
        py::arg("temperature") = 0.7f, 
        py::arg("top_p") = 0.9f, 
        py::arg("repetition_penalty") = 1.1f
        )
        
        .def("next_token", [](InferenceEngine& self, int token, 
                              float temperature, float top_p, float repetition_penalty) {
            return self.next_token(token, temperature, top_p, repetition_penalty);
        }, 
        py::arg("token"), 
        py::arg("temperature") = 0.7f, 
        py::arg("top_p") = 0.9f, 
        py::arg("repetition_penalty") = 1.1f
        )
        .def("nll", &InferenceEngine::nll, "计算损失函数")
        .def_readwrite("im_start_id", &InferenceEngine::im_start_id)
        .def_readwrite("im_end_id", &InferenceEngine::im_end_id)
        .def_readwrite("nl_id", &InferenceEngine::nl_id)
        .def_readwrite("role_id_user", &InferenceEngine::role_id_user)
        .def_readwrite("role_id_assistant", &InferenceEngine::role_id_assistant);
}
