#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cassert>

class Tensor {
public:
    std::shared_ptr<float[]> data_;
    std::vector<int> shape_;
    std::vector<int> strides_;

public:
    Tensor(std::vector<int> shape);
    ~Tensor();

    // 强制浅拷贝，禁用深拷贝
    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor clone() const;

    // 输出
    void print() const;

    // 导入二进制文件
    void load(const std::string& filename) const;

    // 总大小、维度及形状
    int size() const;
    int dims() const;
    std::vector<int> shape() const;

    // 访问元素
    int calc_index(const std::vector<int>& pos) const;
    float& operator[](const std::vector<int>& pos);
    const float& operator[](const std::vector<int>& pos) const;

    // reshape相关操作
    bool is_contiguous() const;
    Tensor contiguous() const;
    Tensor reshape(const std::vector<int>& shape) const;
    Tensor transpose(const std::vector<int>& perm) const;

    // 四则运算
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(float other) const;
    Tensor operator-(float other) const;
    Tensor operator*(float other) const;
    Tensor operator/(float other) const;

    // 自加，用于加速残差链接
    void add(const Tensor& other);
};