#include "tensor.h"
#include "utils.h"
#include "ops.h"
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cassert>
#include <immintrin.h>

const int max_len = 1024;

SwishGLU::SwishGLU(int in_dim, int hidden_dim, std::string device) : 
    W1_(in_dim, hidden_dim, device), 
    W2_(hidden_dim, in_dim, device), 
    V_(in_dim, hidden_dim, device), 
    workplace_W({max_len, hidden_dim}, device),
    workplace_V({max_len, hidden_dim}, device),
    vec_W({hidden_dim}, device),
    vec_V({hidden_dim}, device),
    hidden_dim(hidden_dim),
    device_(device) {
}

SwishGLU::~SwishGLU() {}

void SwishGLU::load(const std::string& base_path) {
    W1_.load(base_path + "/W1");
    W2_.load(base_path + "/W2");
    V_.load(base_path + "/V");
}

std::string SwishGLU::device() const {
    return device_;
}

void SwishGLU::to(const std::string& device) {
    if (device_ == device) return;
    if (device == "cpu") {
        W1_.to("cpu");
        W2_.to("cpu");
        V_.to("cpu");
        workplace_W.to("cpu");
        workplace_V.to("cpu");
        vec_W.to("cpu");
        vec_V.to("cpu");
    }
    else if (device == "cuda") {
        W1_.to("cuda");
        W2_.to("cuda");
        V_.to("cuda");
        workplace_W.to("cuda");
        workplace_V.to("cuda");
        vec_W.to("cuda");
        vec_V.to("cuda");
    }
    else {
        std::cerr << "SwishGLU::to : Invalid device." << std::endl;
        exit(1);
    }
    device_ = device;
}

const int BS = 16;

// 对A做silu后和B逐点相乘，并赋值回A
__device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}
__global__ void silu_mul_kernel(float* A, const float* B, int N, int K) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x < N && idx_y < K) {
        A[idx_x * K + idx_y] = silu(A[idx_x * K + idx_y]) * B[idx_x * K + idx_y];
    }
}

float silu_cpu(float x) {
    return x / (1.0f + std::exp(-x));
}
void silu_mul_cpu(float* A, const float* B, int N, int K) {
    # pragma omp parallel for schedule(static)
    for (int i=0; i<N; i++) {
        for (int j=0; j<K; j++) {
            A[i * K + j] = silu_cpu(A[i * K + j]) * B[i * K + j];
        }
    }
}


// 由于这个kernel是在调用过linear后采用的，我们不需要对它做shape的check了
void SwishGLU::forward(Tensor& x) {
    int N = x.shape_[0];
    int K = x.shape_[1];
    int H = hidden_dim;
    workplace_W.shape_ = {N, H}; W1_.forward(x, workplace_W); 
    workplace_V.shape_ = {N, H}; V_.forward(x, workplace_V);
    
    if (device_ == "cpu"){
        silu_mul_cpu(workplace_W.h_ptr, workplace_V.h_ptr, N, H);
    }
    else {
        dim3 grid((N + BS - 1) / BS, (H + BS - 1) / BS);
        dim3 block(BS, BS);
        silu_mul_kernel<<<grid, block>>>(workplace_W.d_ptr, workplace_V.d_ptr, N, H);
        CHECK_KERNEL_LAUNCH();
    }
    W2_.forward(workplace_W, x);
}

const int BS_vec = 256;
__global__ void silu_mul_vec_kernel(float* A, const float* B, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        A[idx] = silu(A[idx]) * B[idx];
    }
}

void silu_mul_vec_cpu(float* A, const float* B, int K) {
    # pragma omp parallel for schedule(static)
    for (int i=0; i<K; i++) {
        A[i] = silu_cpu(A[i]) * B[i];
    }
}

void SwishGLU::forward_vec(Tensor& x) {
    int K = x.shape_[0];
    W1_.forward_vec(x, vec_W);
    V_.forward_vec(x, vec_V);
    if (device_ == "cpu") {
        silu_mul_vec_cpu(vec_W.h_ptr, vec_V.h_ptr, hidden_dim);
    }
    else {
        dim3 grid((hidden_dim + BS_vec - 1) / BS_vec);
        dim3 block(BS_vec);
        silu_mul_vec_kernel<<<grid, block>>>(vec_W.d_ptr, vec_V.d_ptr, hidden_dim);
        CHECK_KERNEL_LAUNCH();
    }
    W2_.forward_vec(vec_W, x);
}