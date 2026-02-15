#include "tensor.h"
#include "utils.h"
#include "ops.h"
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cassert>
#include <immintrin.h>

const int TILE_SIZE = 16;

Linear::Linear(
    int in_dim, 
    int out_dim, 
    std::string device
):
    weight_({out_dim, in_dim}, device),
    bias_({out_dim}, device),
    weight_T_({in_dim, out_dim}, device),
    device_(device) {
}

Linear::~Linear() {}

// src(M, N) -> dst(N, M)
__global__ void transpose_kernel(const float* src, float* dst, int M, int N) {
    int bx = blockIdx.x * blockDim.x, tx = threadIdx.x;
    int by = blockIdx.y * blockDim.y, ty = threadIdx.y;
    __shared__ float src_s[TILE_SIZE][TILE_SIZE + 1];
    if (by + ty < M && bx + tx < N) {
        src_s[ty][tx] = src[(by + ty) * N + (bx + tx)];
    }
    __syncthreads();
    if (by + tx < M && bx + ty < N) {
        dst[(bx + ty) * M + (by + tx)] = src_s[tx][ty];
    }
}

void transpose(const Tensor& src, Tensor& dst) {
    if (src.shape_.size() != 2 || dst.shape_.size() != 2) {
        std::cerr << "transpose only support 2D tensor" << std::endl;
        exit(1);
    }
    if (src.shape_[0] != dst.shape_[1] || src.shape_[1] != dst.shape_[0]) {
        std::cerr << "transpose shape not match" << std::endl;
        __print_vec__(src.shape_, "src");
        __print_vec__(dst.shape_, "dst");
        exit(1);
    }
    if (src.device_ == "cpu" && dst.device_ == "cpu") {
        # pragma omp parallel for schedule(static)
        for (int i = 0; i < src.shape_[0]; i++) {
            for (int j = 0; j < src.shape_[1]; j++) {
                dst.h_ptr[j * dst.shape_[1] + i] = src.h_ptr[i * src.shape_[1] + j];
            }
        }
    }
    else if (src.device_ == "cuda" && dst.device_ == "cuda") {
        int M = src.shape_[0], N = src.shape_[1];
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        transpose_kernel<<<grid, block>>>(src.d_ptr, dst.d_ptr, M, N);
        CHECK_KERNEL_LAUNCH();
    }
}

void Linear::load(const std::string& path) {
    weight_.load(path + "/weight.bin");
    bias_.load(path + "/bias.bin");
    transpose(weight_, weight_T_);
}

std::string Linear::device() const {
    return device_;
}

void Linear::to(const std::string& device) {
    if (device_ == device) return;
    if (device == "cpu") {
        weight_.to("cpu");
        bias_.to("cpu");
        weight_T_.to("cpu");
    }
    else if (device == "cuda") {
        weight_.to("cuda");
        bias_.to("cuda");
        weight_T_.to("cuda");
    }
    else {
        std::cerr << "Invalid device: " << device << std::endl;
        exit(1);
    }
    device_ = device;
}


// 约定 weight_(M, K)，input(N, K), bias(M), output(N, M)
// 使用 register tiling, 一次处理 4*4 的目标格子
// sum处理的位置相当于 output[y][x], x:0~M, y:0~N
// output[y][x] = bias[x] + sum(weight[x][k] * input[y][k])
// 原模型中任何涉及linear的地方，维度K都是64的倍数，所以不考虑K的尾部效应
__device__ int __min__(int a, int b) {
    return a < b ? a : b;
}
__global__ void linear_forward_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int M, int N, int K
) {
    int bx = blockIdx.x * blockDim.x * 4, tx = threadIdx.x;
    int by = blockIdx.y * blockDim.y * 4, ty = threadIdx.y;
    __shared__ float w_s[TILE_SIZE * 4][TILE_SIZE + 1];
    __shared__ float i_s[TILE_SIZE * 4][TILE_SIZE + 1];
    float sum[4][4] = {0.0f};
    float w_r[4] = {0.0f};
    float i_r[4] = {0.0f};
    for (int i=0; i<K/TILE_SIZE; i++) {
        if (bx + ty * 4 < M) {
            int T = __min__(M - (bx + ty * 4), 4);
            for (int j=0; j<T; j++) {
                w_s[ty * 4 + j][tx] = weight[(bx + ty * 4 + j) * K + i * TILE_SIZE + tx];
            }
        }
        if (by + ty * 4 < N) {
            int T = __min__(N - (by + ty * 4), 4);
            for (int j=0; j<T; j++) {
                i_s[ty * 4 + j][tx] = input[(by + ty * 4 + j) * K + i * TILE_SIZE + tx];
            }
        }
        __syncthreads();
        if (bx + tx * 4 < M && by + ty * 4 < N) {
            for (int j=0; j<TILE_SIZE; j++) {
                w_r[0] = w_s[tx * 4 + 0][j];
                w_r[1] = w_s[tx * 4 + 1][j];
                w_r[2] = w_s[tx * 4 + 2][j];
                w_r[3] = w_s[tx * 4 + 3][j];
                i_r[0] = i_s[ty * 4 + 0][j];
                i_r[1] = i_s[ty * 4 + 1][j];
                i_r[2] = i_s[ty * 4 + 2][j];
                i_r[3] = i_s[ty * 4 + 3][j];
                for (int k=0; k<4; k++) {
                    for (int l=0; l<4; l++) {
                        sum[l][k] += w_r[k] * i_r[l];
                    }
                }
            }
        }
        __syncthreads();
    }
    __shared__ float b_s[TILE_SIZE * 4];
    if (bx + tx * 4 < M) {
        int T = __min__(M - (bx + tx * 4), 4);
        for (int j=0; j<T; j++) {
            b_s[tx * 4 + j] = bias[bx + tx * 4 + j];
        }
    }
    __syncthreads();
    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            if (bx + tx * 4 + i < M && by + ty * 4 + j < N) {
                output[(by + ty * 4 + j) * M + (bx + tx * 4 + i)] = b_s[tx * 4 + i] + sum[j][i];
            }
        }
    }
}

void linear_forward_cpu(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int M, int N, int K
) {
    # pragma omp parallel for collapse(1) schedule(static)
    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            __m256 res_vec = _mm256_setzero_ps();
            for (int k=0; k<K; k+=8) {
                __m256 a_vec = _mm256_loadu_ps(&input[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&weight[j * K + k]);
                res_vec = _mm256_fmadd_ps(a_vec, b_vec, res_vec);
            }
            float res_val[8];
            _mm256_storeu_ps(res_val, res_vec);
            float sum = 0.0;
            for (int k=0; k<8; k++) {
                sum += res_val[k];
            }
            output[i * M + j] = bias[j] + sum;
        }
    }
}

void Linear::forward(const Tensor& input, Tensor& output) {
    bool shape_match = true;
    if (input.dims() == 0 || output.dims() == 0 || input.dims() != output.dims()) {
        shape_match = false;
    }
    if (input.shape_[input.shape_.size() - 1] != weight_.shape_[1]) {
        shape_match = false;
    }
    if (output.shape_[output.shape_.size() - 1] != weight_.shape_[0]) {
        shape_match = false;
    }
    for (int i=0; i<input.shape_.size() - 1; i++) {
        if (input.shape_[i] != output.shape_[i]) {
            shape_match = false;
            break;
        }
    }
    if (!shape_match) {
        std::cerr << "Linear forward: Shape not match: " << std::endl;
        __print_vec__(input.shape_, "input_shape");
        __print_vec__(output.shape_, "output_shape");
        __print_vec__(weight_.shape_, "weight_shape");
        exit(1);
    }
    int K = input.shape_[input.shape_.size() - 1];
    int M = weight_.shape_[0];
    int N = input.size() / K;
    if (input.device() == "cpu" && output.device() == "cpu" && device_ == "cpu") {
        linear_forward_cpu(
            input.h_ptr,
            weight_.h_ptr,
            bias_.h_ptr,
            output.h_ptr,
            M, N, K
        );
    }
    else if (input.device() == "cuda" && output.device() == "cuda" && device_ == "cuda") {
        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 grid_size((weight_.shape_[0] + block_size.x * 4 - 1) / (block_size.x * 4), 
                       (input.shape_[0] + block_size.y * 4 - 1) / (block_size.y * 4));
        linear_forward_kernel<<<grid_size, block_size>>>(
            input.d_ptr, 
            weight_.d_ptr, 
            bias_.d_ptr, 
            output.d_ptr, 
            M, N, K
        );
        CHECK_KERNEL_LAUNCH();
    }
    else {
        std::cerr << "Linear forward: devices not matched: " << std::endl;
        std::cerr << "input: " << input.device() << std::endl;
        std::cerr << "output: " << output.device() << std::endl;
        std::cerr << "linear: " << device_ << std::endl;
        exit(1);
    }
}

// 仅支持对向量(1-dim tensor)变换
// 即 input(K), weight_T(K, M), bias(M), output(M)
// output[x] = bias[x] + sum(weight_T[k][x] * input[k])
// 原模型中任何涉及linear的地方，维度K都是256的倍数，所以仍不考虑K的尾部效应
const int BLOCK_DIM = 256;
__global__ void linear_forward_vec_kernel(
    const float* input, 
    const float* weight_T, 
    const float* bias, 
    float* output, 
    int M, int K
) {
    int bx = blockIdx.x * blockDim.x, tx = threadIdx.x;
    __shared__ float i_s[BLOCK_DIM];
    float sum = 0.0f;
    for (int i=0; i<K / BLOCK_DIM; i++) {
        i_s[tx] = input[i * BLOCK_DIM + tx];
        __syncthreads();
        for (int j=0; j<BLOCK_DIM; j++) {
        //    sum += weight_T[(i * BLOCK_DIM + j) + (bx + tx) * K] * i_s[j];
            sum += weight_T[(i * BLOCK_DIM + j) * M + bx + tx] * i_s[j];
        }
        __syncthreads();
    }
    if (bx + tx < M) {
        output[bx + tx] = bias[bx + tx] + sum;
    }
}

void linear_forward_vec_cpu(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int M, int K
) {
    # pragma omp parallel for schedule(static)
    for (int i=0; i<M; i++) {
        __m256 res_vec = _mm256_setzero_ps();
        for (int k=0; k<K; k+=8) {
            __m256 a_vec = _mm256_loadu_ps(&input[k]);
            __m256 b_vec = _mm256_loadu_ps(&weight[i * K + k]);
            res_vec = _mm256_fmadd_ps(a_vec, b_vec, res_vec);
        }
        float res_val[8];
        _mm256_storeu_ps(res_val, res_vec);
        float sum = 0.0;
        for (int k=0; k<8; k++) {
            sum += res_val[k];
        }
        output[i] = bias[i] + sum;
    }
}

void Linear::forward_vec(const Tensor& input, Tensor& output) {
    bool shape_match = true;
    if (input.dims() != 1 || output.dims() != 1) {
        shape_match = false;
    }
    if (input.shape_[0] != weight_.shape_[1]) {
        shape_match = false;
    }
    if (output.shape_[0] != weight_.shape_[0]) {
        shape_match = false;
    }
    if (!shape_match) {
        std::cerr << "Linear forward_vec: Shape not match: " << std::endl;
        __print_vec__(input.shape_, "input_shape");
        __print_vec__(output.shape_, "output_shape");
        __print_vec__(weight_.shape_, "weight_shape");
        exit(1);
    }
    if (input.device() == "cpu" && output.device() == "cpu" && device_ == "cpu") {
        linear_forward_vec_cpu(
            input.h_ptr,
            weight_.h_ptr,
            bias_.h_ptr,
            output.h_ptr,
            weight_.shape_[0],
            weight_.shape_[1]
        );
    }
    else if (input.device() == "cuda" && output.device() == "cuda" && device_ == "cuda") {
        dim3 block_size(BLOCK_DIM);
        dim3 grid_size((weight_.shape_[0] + block_size.x - 1) / block_size.x);
        linear_forward_vec_kernel<<<grid_size, block_size>>>(
            input.d_ptr, 
            weight_T_.d_ptr, 
            bias_.d_ptr, 
            output.d_ptr, 
            weight_.shape_[0], 
            weight_.shape_[1]
        );
        CHECK_KERNEL_LAUNCH();
    }
    else {
        std::cerr << "Linear forward: devices not matched: " << std::endl;
        std::cerr << "input: " << input.device() << std::endl;
        std::cerr << "output: " << output.device() << std::endl;
        std::cerr << "linear: " << device_ << std::endl;
        exit(1);
    }
}