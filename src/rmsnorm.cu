#include "tensor.h"
#include "utils.h"
#include "ops.h"
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cassert>
#include <immintrin.h>

const int BLOCK_SIZE = 256;

RMSNorm::RMSNorm(int dim, std::string device):
    gamma_({dim}, device),
    device_(device) {
}

RMSNorm::~RMSNorm() {}

void RMSNorm::load(const std::string& path) {
    gamma_.load(path + "/weight.bin");
}

std::string RMSNorm::device() const {
    return device_;
}

void RMSNorm::to(const std::string& device) {
    if (device_ == device) return;
    if (device == "cpu") {
        gamma_.to("cpu");
    }
    else if (device == "cuda") {
        gamma_.to("cuda");
    }
    else {
        std::cerr << "RMSNorm : Invalid device: " << device << std::endl;
        exit(1);
    }
    device_ = device;
}

// input(N, K) -> output(N, K)
// gamma(K)
// 一个block处理一个K
__global__ void rmsnorm_kernel(
    float* input, 
    float* gamma, 
    int N, int K
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    int BLO = (K + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    __shared__ float bs[BLOCK_SIZE];
    bs[tx] = 0.0f;
    for (int i=0; i<BLO; i++) {
        float tmp0 = 0.0f, tmp1 = 0.0f;
        if (BLOCK_SIZE * (2 * i + 0) + tx < K) {
            tmp0 = input[bx * K + BLOCK_SIZE * (2 * i + 0) + tx];
        }
        if (BLOCK_SIZE * (2 * i + 1) + tx < K) {
            tmp1 = input[bx * K + BLOCK_SIZE * (2 * i + 1) + tx];
        }
        bs[tx] += tmp0 * tmp0 + tmp1 * tmp1;
    }
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            bs[tx] += bs[tx + s];
        }
        __syncthreads();
    }
    float inv_sqrt_mean = rsqrt(bs[0] / K + 1e-6f);
    int BLO2 = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i=0; i<BLO2; i++) {
        if (BLOCK_SIZE * i + tx < K) {
            float coef = inv_sqrt_mean * gamma[BLOCK_SIZE * i + tx];
            input[bx * K + BLOCK_SIZE * i + tx] *= coef;
        }
    }
}

void rmsnorm_cpu(
    float* input, 
    float* gamma, 
    int N, int K
) {
    # pragma omp parallel for schedule(static)
    for (int i=0; i<N; i++) {
        __m256 sum_vec = _mm256_setzero_ps();
        for (int j=0; j<K; j+=8) {
            __m256 x_vec = _mm256_loadu_ps(&input[i * K + j]);
            sum_vec = _mm256_fmadd_ps(x_vec, x_vec, sum_vec);
        }
        float sum_val[8];
        _mm256_storeu_ps(sum_val, sum_vec);
        float sum = 0.0;
        for (int k=0; k<8; k++) {
            sum += sum_val[k];
        }
        float sqrt_mean = std::sqrt(sum / K + 1e-6);
        float inv_sqrt_mean = 1.0 / sqrt_mean;
        for (int j=0; j<K; j+=8) {
            __m256 x_vec = _mm256_loadu_ps(&input[i * K + j]);
            __m256 gamma_vec = _mm256_loadu_ps(&gamma[j]);
            x_vec = _mm256_mul_ps(x_vec, _mm256_set1_ps(inv_sqrt_mean));
            x_vec = _mm256_mul_ps(x_vec, gamma_vec);
            _mm256_storeu_ps(&input[i * K + j], x_vec);
        }
    }
}

void RMSNorm::forward(Tensor& input) {
    if (input.shape_.empty()) {
        std::cerr << "warning: RMSNorm : input tensor is empty" << std::endl;
        return;
    }
    int K = input.shape_[input.shape_.size() - 1];
    int N = input.size() / K;
    if (K != gamma_.shape_[0]) {
        std::cerr << "RMSNorm : input and gamma shape not match" << std::endl;
        __print_vec__(input.shape_, "input");
        __print_vec__(gamma_.shape_, "gamma");
        exit(1);
    }
    if (input.device_ == "cpu" && gamma_.device_ == "cpu") {
        rmsnorm_cpu(
            input.h_ptr,
            gamma_.h_ptr,
            N, K
        );
    }
    else if (input.device_ == "cuda" && gamma_.device_ == "cuda") {
        dim3 grid(N);
        dim3 block(BLOCK_SIZE);
        rmsnorm_kernel<<<grid, block>>>(
            input.d_ptr,
            gamma_.d_ptr,
            N, K
        );
        CHECK_CUDA(cudaGetLastError());
    }
    else {
        std::cerr << "RMSNorm : devices not matched: " << std::endl;
        std::cerr << "input: " << input.device_ << std::endl;
        std::cerr << "gamma: " << gamma_.device_ << std::endl;
        exit(1);
    }
}