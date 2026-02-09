#include "tensor.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <memory>
#include <cassert>

Tensor::Tensor(std::vector<int> shape) {
    int size = 1;
    for (int i = 0; i < shape.size(); i++) {
        assert((shape[i] > 0));
        size *= shape[i];
    }
    data_ = std::shared_ptr<float[]>(new float[size]);
    shape_ = shape;
    strides_ = std::vector<int>(shape.size());
    for (int i = shape.size() - 1; i >= 0; i--) {
        if (i == shape.size() - 1) strides_[i] = 1;
        else strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

Tensor::~Tensor() {
    // 用了shared_ptr，不需要析构
}

Tensor Tensor::clone() const {
    Tensor res(shape_);
    std::copy(data_.get(), data_.get() + size(), res.data_.get());
    return res;
}

void Tensor::print() const {
    std::cout << "Tensor, shape: [";
    for (int i = 0; i < shape_.size(); i++) {
        std::cout << shape_[i];
        if (i != shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void Tensor::load(const std::string& filename) const {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("打开文件失败: " + filename);

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if(!file.read(reinterpret_cast<char*>(data_.get()), size() * sizeof(float)))
        throw std::runtime_error("读取文件失败: " + filename);
    file.close();
}

int Tensor::size() const {
    int size = 1;
    for (int i = 0; i < shape_.size(); i++) {
        size *= shape_[i];
    }
    return size;
}

int Tensor::dims() const {
    return shape_.size();
}

std::vector<int> Tensor::shape() const {
    return shape_;
}

int Tensor::calc_index(const std::vector<int>& pos) const {
    int index = 0;
    for (int i = 0; i < pos.size(); i++) {
        index += pos[i] * strides_[i];
    }
    return index;
}

float& Tensor::operator[](const std::vector<int>& pos) {
    return data_[calc_index(pos)];
}

const float& Tensor::operator[](const std::vector<int>& pos) const {
    return data_[calc_index(pos)];
}

bool Tensor::is_contiguous() const {
    int size_ = 1;
    for (int i = 0; i < shape_.size(); i++) {
        size_ *= shape_[i];
        if (strides_[i] * size_ != size()) return false;
    }
    return true;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    Tensor res(shape_);
    std::vector<int> pos(shape_.size(), 0);
    int index = 0, size_ = size();
    for (int i = 0; i < size_; i++) {
        res.data_[i] = data_[index];
        pos[shape_.size() - 1] ++;
        index += strides_[shape_.size() - 1];
        for (int j = shape_.size() - 1; j > 0; j--) {
            if (pos[j] == shape_[j]) {
                index -= pos[j] * strides_[j];
                pos[j] = 0;
                index += strides_[j - 1];
                pos[j - 1] ++;
            }
            else break;
        }
    }
    return res;
}

Tensor Tensor::reshape(const std::vector<int>& shape) const {
    Tensor res = contiguous();
    res.shape_ = shape;
    res.strides_ = std::vector<int>(shape.size());
    if (shape.size() == 0) {
        return res;
    }
    assert((res.size() == size()));
    res.strides_[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) {
        res.strides_[i] = res.strides_[i + 1] * shape[i + 1];
    }
    return res;
}

Tensor Tensor::transpose(const std::vector<int>& perm) const {
    Tensor res = *this;
    if (perm.size() != res.shape_.size()) {
        throw std::runtime_error("transpose: 维度不匹配");
    }
    for (int i = 0; i < perm.size(); i++) {
        res.shape_[i] = shape_[perm[i]];
        res.strides_[i] = strides_[perm[i]];
    }
    return res;
}

void _broadcast(Tensor& A, Tensor& B) {
    if (A.shape_.size() < B.shape_.size()) {
        throw std::runtime_error("broadcast: 维度不匹配");
    }
    for (int i = 0; i < B.shape_.size(); i++) {
        if (A.shape_[i] != B.shape_[i]) {
            if (A.shape_[i] == 1) {
                A.shape_[i] = B.shape_[i];
                A.strides_[i] = 0;
            }
            else if (B.shape_[i] == 1) {
                B.shape_[i] = A.shape_[i];
                B.strides_[i] = 0;
            }
            else {
                throw std::runtime_error("broadcast: 维度不匹配");
            }
        }
    }
}

// 支持广播机制
Tensor Tensor::operator+(const Tensor& other) const {
    Tensor A = this -> contiguous();
    Tensor B = other.contiguous();
    if (A.shape_ == B.shape_) {
        Tensor res(A.shape_);
        for (int i = 0; i < A.size(); i++) {
            res.data_[i] = A.data_[i] + B.data_[i];
        }
        return res;
    }
    else {
        _broadcast(A, B);
        Tensor res(A.shape_);
        std::vector<int> pos(A.shape_.size(), 0);
        int index_A = 0, index_B = 0;
        for (int i = 0; i < A.size(); i++) {
            res.data_[i] = A.data_[index_A] + B.data_[index_B];
            pos[A.shape_.size() - 1] ++;
            index_A += A.strides_[A.shape_.size() - 1];
            index_B += B.strides_[B.shape_.size() - 1];
            for (int j = A.shape_.size() - 1; j > 0; j--) {
                if (pos[j] == A.shape_[j]) {
                    index_A -= pos[j] * A.strides_[j];
                    index_B -= pos[j] * B.strides_[j];
                    pos[j] = 0;
                    index_A += A.strides_[j - 1];
                    index_B += B.strides_[j - 1];
                    pos[j - 1] ++;
                }
                else break;
            }
        }
        return res;
    }
}

Tensor Tensor::operator-(const Tensor& other) const {
    Tensor A = this -> contiguous();
    Tensor B = other.contiguous();
    if (A.shape_ == B.shape_) {
        Tensor res(A.shape_);
        for (int i = 0; i < A.size(); i++) {
            res.data_[i] = A.data_[i] - B.data_[i];
        }
        return res;
    }
    else {
        _broadcast(A, B);
        Tensor res(A.shape_);
        std::vector<int> pos(A.shape_.size(), 0);
        int index_A = 0, index_B = 0;
        for (int i = 0; i < A.size(); i++) {
            res.data_[i] = A.data_[index_A] - B.data_[index_B];
            pos[A.shape_.size() - 1] ++;
            index_A += A.strides_[A.shape_.size() - 1];
            index_B += B.strides_[B.shape_.size() - 1];
            for (int j = A.shape_.size() - 1; j > 0; j--) {
                if (pos[j] == A.shape_[j]) {
                    index_A -= pos[j] * A.strides_[j];
                    index_B -= pos[j] * B.strides_[j];
                    pos[j] = 0;
                    index_A += A.strides_[j - 1];
                    index_B += B.strides_[j - 1];
                    pos[j - 1] ++;
                }
                else break;
            }
        }
        return res;
    }
    
}

Tensor Tensor::operator*(const Tensor& other) const {
    Tensor A = this -> contiguous();
    Tensor B = other.contiguous();
    if (A.shape_ == B.shape_) {
        Tensor res(A.shape_);
        for (int i = 0; i < A.size(); i++) {
            res.data_[i] = A.data_[i] * B.data_[i];
        }
        return res;
    }
    else {
        _broadcast(A, B);
        Tensor res(A.shape_);
        std::vector<int> pos(A.shape_.size(), 0);
        int index_A = 0, index_B = 0;
        for (int i = 0; i < A.size(); i++) {
            res.data_[i] = A.data_[index_A] * B.data_[index_B];
            pos[A.shape_.size() - 1] ++;
            index_A += A.strides_[A.shape_.size() - 1];
            index_B += B.strides_[B.shape_.size() - 1];
            for (int j = A.shape_.size() - 1; j > 0; j--) {
                if (pos[j] == A.shape_[j]) {
                    index_A -= pos[j] * A.strides_[j];
                    index_B -= pos[j] * B.strides_[j];
                    pos[j] = 0;
                    index_A += A.strides_[j - 1];
                    index_B += B.strides_[j - 1];
                    pos[j - 1] ++;
                }
                else break;
            }
        }
        return res;
    }
    
}

Tensor Tensor::operator/(const Tensor& other) const {
    Tensor A = this -> contiguous();
    Tensor B = other.contiguous();
    if (A.shape_ == B.shape_) {
        Tensor res(A.shape_);
        for (int i = 0; i < A.size(); i++) {
            res.data_[i] = A.data_[i] / B.data_[i];
        }
        return res;
    }
    else {
        _broadcast(A, B);
        Tensor res(A.shape_);
        std::vector<int> pos(A.shape_.size(), 0);
        int index_A = 0, index_B = 0;
        for (int i = 0; i < A.size(); i++) {
            res.data_[i] = A.data_[index_A] / B.data_[index_B];
            pos[A.shape_.size() - 1] ++;
            index_A += A.strides_[A.shape_.size() - 1];
            index_B += B.strides_[B.shape_.size() - 1];
            for (int j = A.shape_.size() - 1; j > 0; j--) {
                if (pos[j] == A.shape_[j]) {
                    index_A -= pos[j] * A.strides_[j];
                    index_B -= pos[j] * B.strides_[j];
                    pos[j] = 0;
                    index_A += A.strides_[j - 1];
                    index_B += B.strides_[j - 1];
                    pos[j - 1] ++;
                }
                else break;
            }
        }
        return res;
    }
}


Tensor Tensor::operator+(float other) const {
    Tensor res(shape_);
    for (int i = 0; i < size(); i++) {
        res.data_[i] = data_[i] + other;
    }
    return res;
}   

Tensor Tensor::operator-(float other) const {
    Tensor res(shape_);
    for (int i = 0; i < size(); i++) {
        res.data_[i] = data_[i] - other;
    }
    return res;
}

Tensor Tensor::operator*(float other) const {
    Tensor res(shape_);
    for (int i = 0; i < size(); i++) {
        res.data_[i] = data_[i] * other;
    }
    return res;
}

Tensor Tensor::operator/(float other) const {
    Tensor res(shape_);
    for (int i = 0; i < size(); i++) {
        res.data_[i] = data_[i] / other;
    }
    return res;
}

void Tensor::add(const Tensor& other) {
    Tensor o_c = other.contiguous();
    if (shape_ != o_c.shape_) {
        throw std::runtime_error("add: 维度不匹配");
    }
    # pragma omp parallel for
    for (int i = 0; i < size(); i++) {
        data_[i] += o_c.data_[i];
    }
}
