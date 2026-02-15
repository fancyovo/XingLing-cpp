#include "tensor.h"
#include "utils.h"
#include "ops.h"
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cassert>
#include <immintrin.h>

const int BS_C = 32; 
const int BS_m = 8;

// rope的对象一定是(..., m, K/m)的形式
// (N, m, K/m = C)
// 考虑一个block处理 BLOCK_SIZE 个注意力头
// ...和m/BLOCK_SIZE交给grid
// tx 直接处理一个 head，ty处理不同head
// by也处理不同的head，bx直接对应len
__global__ void rope_kernel(
    float* input,
    float base,
    int N, int m, int C, 
    int start_pos
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    int by = blockIdx.y, ty = threadIdx.y;
    if (by * BS_m + ty < m) {
        int idx = bx * m * C + (by * BS_m + ty) * C;
        for (int i=0; i<(C/2 + BS_C - 1) / BS_C; i++) {
            int j = i * BS_C + tx;
            if (j < C / 2) {
                float theta = std::pow(base, -2.0f * j / C) * (bx + start_pos);
                float cosP = std::cos(theta);
                float sinP = std::sin(theta);
                float i0 = input[idx + j];
                float i1 = input[idx + j + C/2];
                float o0 =  i0 * cosP + i1 * sinP;
                float o1 = -i0 * sinP + i1 * cosP;
                input[idx + j] = o0;
                input[idx + j + C/2] = o1;
            }
        }
    }
}

void rope_cpu(
    float* input,
    float base,
    int N, int m, int C
) {
    # pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<N; i++) {
        for (int j=0; j<m; j++) {
            for (int k = 0; k < C / 2; k++) {
                float theta = std::pow(base, -2.0f * k / C) * i;
                float cosP = std::cos(theta);
                float sinP = std::sin(theta);
                float i0 = input[i * m * C + j * C + k];
                float i1 = input[i * m * C + j * C + k + C/2];
                float o0 =  i0 * cosP + i1 * sinP;
                float o1 = -i0 * sinP + i1 * cosP;
                input[i * m * C + j * C + k] = o0;
                input[i * m * C + j * C + k + C/2] = o1;
            }
        }
    }
}

void RoPE::forward(Tensor& x, float base) {
    if (x.shape_.size() != 3) {
        std::cerr << "RoPE: input shape error" << std::endl;
        __print_vec__(x.shape_, "x");
        exit(1);
    }
    int N = x.shape_[0];
    int m = x.shape_[1];
    int C = x.shape_[2];
    if (C % 2 != 0) {
        std::cerr << "RoPE: dim must be even" << std::endl;
        __print_vec__(x.shape_, "x");
        exit(1);
    }
    if (x.device_ == "cpu") {
        rope_cpu(
            x.h_ptr,
            base,
            N, m, C
        );
    }
    else if (x.device_ == "cuda") {
        dim3 grid(N, (m + BS_m - 1) / BS_m);
        dim3 block(BS_C, BS_m);
        rope_kernel<<<grid, block>>>(
            x.d_ptr,
            base,
            N, m, C,
            0
        );
        CHECK_KERNEL_LAUNCH();
    }
    else {
        std::cerr << "RoPE: Invalid device: " << x.device_ << std::endl;
        exit(1);
    }
}

void rope_vec_cpu(
    float* input,
    float base,
    int N, int m, int C, 
    int pos
) {
    # pragma omp parallel for schedule(static)
    for (int i=0; i<m; i++) {
        for (int j = 0; j < C / 2; j++) {
            float theta = std::pow(base, -2.0f * j / C) * pos;
            float cosP = std::cos(theta);
            float sinP = std::sin(theta);
            float i0 = input[i * C + j];
            float i1 = input[i * C + j + C/2];
            float o0 =  i0 * cosP + i1 * sinP;
            float o1 = -i0 * sinP + i1 * cosP;
            input[i * C + j] = o0;
            input[i * C + j + C/2] = o1;
        }
    }
}

void RoPE::forward_vec(Tensor& x, int pos, float base) {
    if (x.shape_.size() != 2) {
        std::cerr << "RoPE: input shape error" << std::endl;
        __print_vec__(x.shape_, "x");
        exit(1);
    }
    int m = x.shape_[0];
    int C = x.shape_[1];
    if (C % 2 != 0) {
        std::cerr << "RoPE: dim must be even" << std::endl;
        __print_vec__(x.shape_, "x");
        exit(1);
    }
    if (x.device_ == "cpu") {
        rope_vec_cpu(
            x.h_ptr,
            base,
            1, m, C,
            pos
        );
    }
    else if (x.device_ == "cuda") {
        dim3 grid(1, (m + BS_m - 1) / BS_m);
        dim3 block(BS_C, BS_m);
        rope_kernel<<<grid, block>>>(
            x.d_ptr,
            base,
            1, m, C,
            pos
        );
        CHECK_KERNEL_LAUNCH();
    }
    else {
        std::cerr << "RoPE: Invalid device: " << x.device_ << std::endl;
        exit(1);
    }
}