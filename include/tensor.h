#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cassert>

// 简单起见，不实现transpose且禁用任何拷贝。
// 同时直接暴露data，不提供下标访问接口
class Tensor {
public:
    float* h_ptr;
    float* d_ptr;
    std::vector<int> shape_;
    std::string device_;
    bool cuda_avaliable;

public:
    Tensor(std::vector<int> shape, std::string device);
    ~Tensor();
    // 禁用任何拷贝
    Tensor(const Tensor& other) = delete;
    Tensor& operator=(const Tensor& other) = delete;
    // 设备信息
    void to(const std::string& device);
    std::string device() const;
    // 基本信息
    int size() const;
    int dims() const;
    // 基本构造
    void reshape(const std::vector<int>& shape);

    // 加载数据 
    void load(const std::string& path);
}; 