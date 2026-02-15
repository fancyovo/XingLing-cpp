#include "tensor.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cassert>
#include <fstream>



Tensor::Tensor(std::vector<int> shape, std::string device) {
    for (int i = 0; i < shape.size(); i++) {
        if (shape[i] < 0) {
            std::cerr << "Invalid shape: ";
            __print_vec__(shape, "shape");
            exit(1);
        }
    }
    shape_ = shape;
    device_ = device;
    h_ptr = new float[size()];
    if (checkGpuSupport()) {
        CHECK_CUDA(cudaMalloc(&d_ptr, (size()) * sizeof(float)));
        cuda_avaliable = true;
    } 
    else {
    //    std::cerr << "Warning: Cuda malloc failed: " << e.what() << std::endl;
        cuda_avaliable = false;
    }
    if (device_ == "cpu") {
        memset(h_ptr, 0, (size()) * sizeof(float));
    }
    else if (device_ == "cuda") {
        if (cuda_avaliable) {
            CHECK_CUDA(cudaMemset(d_ptr, 0, (size()) * sizeof(float)));
        }
        else {
            std::cerr << "Cuda malloc failed, cannot set to 0" << std::endl;
            exit(1);
        }
    }
    else {
        std::cerr << "Invalid device: " << device_ << std::endl;
        exit(1);
    }
}
Tensor::~Tensor() {
    delete[] h_ptr;
    if (cuda_avaliable) {
        CHECK_CUDA(cudaFree(d_ptr));
    }
}

std::string Tensor::device() const {
    return device_;
}

void Tensor::to(const std::string& device) {
    if (device_ == device) {
        return;
    }
    if (device == "cpu") {
        CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, size() * sizeof(float), cudaMemcpyDeviceToHost));
        device_ = "cpu";
    }
    else if (device == "cuda") {
        if (cuda_avaliable) {
            CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        else {
            std::cerr << "Cuda malloc failed, cannot copy to cuda" << std::endl;
            exit(1);
        }
        device_ = "cuda";
    }
    else {
        std::cerr << "Invalid device: " << device << std::endl;
        exit(1);
    }
}

int Tensor::size() const {
    int res = 1;
    for (int i = 0; i < shape_.size(); i++) {
        res *= shape_[i];
    }
    return res;
}
int Tensor::dims() const {
    return shape_.size();
}

void Tensor::reshape(const std::vector<int>& shape) {
    int new_size = 1;
    for (int i = 0; i < shape.size(); i++) {
        if (shape[i] < 0) {
            std::cerr << "Invalid shape: ";
            __print_vec__(shape, "shape");
            exit(1);
        }
        new_size *= shape[i];
    }
    if (new_size != size()) {
        std::cerr << "shape not matched: ";
        __print_vec__(shape_, "old_shape");
        __print_vec__(shape, "new_shape");
        exit(1);
    }
    shape_ = shape;
}

void Tensor::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("打开文件失败: " + path);

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if(!file.read(reinterpret_cast<char*>(h_ptr), size() * sizeof(float)))
        throw std::runtime_error("读取文件失败: " + path);
    file.close();
    if (device_ == "cuda") {
        CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, size() * sizeof(float), cudaMemcpyHostToDevice));
    }
}