#pragma once
#include <vector>
#include <string>
#include <random>
#include <functional>
#include <chrono>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>

#include "tensor.h"
#include "ops.h"
#include "utils.h"

// ====== 固定超参数 ======
constexpr int MAX_SEQ_LEN = 768;
constexpr int LATENT_DIM  = 1536;
constexpr int FFN_DIM     = 4096;
constexpr int NUM_HEADS   = 24;
constexpr int NUM_LAYERS  = 16;
constexpr float ROPE_BASE = 10000.0f;
constexpr int VOCAB_SIZE  = 151669;
constexpr int HEAD_DIM    = LATENT_DIM / NUM_HEADS;

// ====== 测试统计 ======
struct TestStats {
    int passed = 0;
    int failed = 0;
    int skipped = 0;
};
extern TestStats g_stats;

// ====== 工具函数 ======
inline bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

inline void test_pass(const std::string& msg) {
    g_stats.passed++;
    std::cout << "[PASS] " << msg << std::endl;
}
inline void test_fail(const std::string& msg) {
    g_stats.failed++;
    std::cout << "[FAIL] " << msg << std::endl;
}
inline void test_skip(const std::string& msg) {
    g_stats.skipped++;
    std::cout << "[SKIP] " << msg << std::endl;
}

inline void expect_true(bool cond, const std::string& msg) {
    if (cond) test_pass(msg);
    else test_fail(msg);
}

inline void expect_allclose(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float atol, float rtol,
    const std::string& msg
){
    if (a.size() != b.size()) {
        test_fail(msg + " (size mismatch)");
        return;
    }
    float max_abs = 0.f, max_rel = 0.f;
    int max_idx = -1;
    bool ok = true;
    for (size_t i=0;i<a.size();++i){
        float diff = std::abs(a[i]-b[i]);
        float denom = std::max(std::abs(a[i]), std::abs(b[i]));
        float tol = atol + rtol * denom;
        if (diff > tol) ok = false;
        if (diff > max_abs) { max_abs = diff; max_idx = (int)i; }
        float rel = diff / (denom + 1e-6f);
        if (rel > max_rel) max_rel = rel;
    }
    if (ok) {
        test_pass(msg);
    } else {
        test_fail(msg + " (max_abs=" + std::to_string(max_abs) +
                  ", max_rel=" + std::to_string(max_rel) +
                  ", idx=" + std::to_string(max_idx) + ")");
    }
}

inline void fill_random(Tensor& t, std::mt19937& rng, float scale=1.0f) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (int i=0;i<t.size();++i) t.h_ptr[i] = dist(rng);
    if (t.device_ == "cuda") {
        CHECK_CUDA(cudaMemcpy(t.d_ptr, t.h_ptr, t.size()*sizeof(float), cudaMemcpyHostToDevice));
    }
}

inline void fill_random_vec(std::vector<float>& v, std::mt19937& rng, float scale=1.0f) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& x : v) x = dist(rng);
}

inline void fill_random_int(std::vector<int>& v, int low, int high, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(low, high);
    for (auto& x : v) x = dist(rng);
}

inline std::vector<float> copy_tensor_to_host(const Tensor& t) {
    std::vector<float> out(t.size());
    if (t.device_ == "cpu") {
        std::memcpy(out.data(), t.h_ptr, t.size()*sizeof(float));
    } else {
        CHECK_CUDA(cudaMemcpy(out.data(), t.d_ptr, t.size()*sizeof(float), cudaMemcpyDeviceToHost));
    }
    return out;
}

// 仅用于：src.h_ptr -> dst.h_ptr，再同步 dst.d_ptr
inline void copy_tensor_data(const Tensor& src, Tensor& dst) {
    if (src.size() != dst.size()) {
        std::cerr << "copy_tensor_data size mismatch" << std::endl;
        exit(1);
    }
    std::memcpy(dst.h_ptr, src.h_ptr, src.size()*sizeof(float));
    if (dst.device_ == "cuda") {
        CHECK_CUDA(cudaMemcpy(dst.d_ptr, dst.h_ptr, dst.size()*sizeof(float), cudaMemcpyHostToDevice));
    }
}

inline void compute_transpose(const Tensor& weight, Tensor& weight_T) {
    int M = weight.shape_[0];
    int K = weight.shape_[1];
    for (int i=0;i<M;i++) {
        for (int k=0;k<K;k++) {
            weight_T.h_ptr[k*M + i] = weight.h_ptr[i*K + k];
        }
    }
}

// 线性层初始化 + weight_T 同步
inline void init_linear_weights(Linear& lin, std::mt19937& rng, float scale=0.02f) {
    fill_random(lin.weight_, rng, scale);
    fill_random(lin.bias_, rng, scale);
    compute_transpose(lin.weight_, lin.weight_T_);
    if (lin.weight_.device_ == "cuda") {
        CHECK_CUDA(cudaMemcpy(lin.weight_.d_ptr, lin.weight_.h_ptr,
                              lin.weight_.size()*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(lin.bias_.d_ptr, lin.bias_.h_ptr,
                              lin.bias_.size()*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(lin.weight_T_.d_ptr, lin.weight_T_.h_ptr,
                              lin.weight_T_.size()*sizeof(float), cudaMemcpyHostToDevice));
    }
}

inline void copy_linear_weights(const Linear& src, Linear& dst) {
    copy_tensor_data(src.weight_, dst.weight_);
    copy_tensor_data(src.bias_, dst.bias_);
    copy_tensor_data(src.weight_T_, dst.weight_T_);
}

// RMSNorm gamma
inline void init_rmsnorm(RMSNorm& norm, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.8f, 1.2f);
    for (int i=0;i<norm.gamma_.size();++i) norm.gamma_.h_ptr[i] = dist(rng);
    if (norm.gamma_.device_ == "cuda") {
        CHECK_CUDA(cudaMemcpy(norm.gamma_.d_ptr, norm.gamma_.h_ptr,
                              norm.gamma_.size()*sizeof(float), cudaMemcpyHostToDevice));
    }
}

inline void copy_rmsnorm(const RMSNorm& src, RMSNorm& dst) {
    copy_tensor_data(src.gamma_, dst.gamma_);
}

// Embedding 初始化
inline void init_embedding(Embedding& emb, std::mt19937& rng, float scale=0.02f) {
    fill_random(emb.weight_, rng, scale);
    if (emb.weight_.device_ == "cuda") {
        CHECK_CUDA(cudaMemcpy(emb.weight_.d_ptr, emb.weight_.h_ptr,
                              emb.weight_.size()*sizeof(float), cudaMemcpyHostToDevice));
    }
}

inline void copy_embedding(const Embedding& src, Embedding& dst) {
    copy_tensor_data(src.weight_, dst.weight_);
}

// ====== 计时 ======
inline float time_cpu_ms(const std::function<void()>& fn, int warmup=2, int repeat=5) {
    for (int i=0;i<warmup;i++) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i=0;i<repeat;i++) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return float(ms / repeat);
}

inline float time_cuda_ms(const std::function<void()>& fn, int warmup=2, int repeat=10) {
    for (int i=0;i<warmup;i++) fn();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i=0;i<repeat;i++) fn();

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms=0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / repeat;
}

inline void print_speed(const std::string& name, const std::string& device,
                        const std::string& phase, int len, float ms) {
    std::cout << "[SPEED] " << name << " " << device
              << " " << phase << " len=" << len
              << " time=" << ms << " ms" << std::endl;
}

// ====== 测试函数原型 ======
void test_linear(bool do_correct, bool do_speed);
void test_rmsnorm(bool do_correct, bool do_speed);
void test_rope(bool do_correct, bool do_speed);
void test_attn(bool do_correct, bool do_speed);
void test_embedding(bool do_correct, bool do_speed);
void test_attentionblock(bool do_correct, bool do_speed);
void test_swishglu(bool do_correct, bool do_speed);
void test_transformerblock(bool do_correct, bool do_speed);
void test_transformer(bool do_correct, bool do_speed, bool do_weight);
