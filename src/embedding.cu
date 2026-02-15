#include "tensor.h"
#include "utils.h"
#include "ops.h"
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cassert>
#include <immintrin.h>



Embedding::Embedding(int vocabulary, int latent_dim, const std::string& device):
    weight_({vocabulary, latent_dim}, device),
    device_(device) {
    if (checkGpuSupport()) {
        CHECK_CUDA(cudaMalloc(&ids_d, 2048 * sizeof(int)));
    } else {
        ids_d = nullptr;
    }
    ids_h = new int[2048];
}

Embedding::~Embedding() {
    CHECK_CUDA(cudaFree(ids_d));
    delete[] ids_h;
}

void Embedding::load(const std::string& base_path) {
    weight_.load(base_path + "/weight.bin");
}

void Embedding::to(const std::string& device) {
    if (device_ == device) return;
    if (device == "cpu") {
        weight_.to("cpu");
    }
    else if (device == "cuda") {
        weight_.to("cuda");
    }
    else {
        std::cerr << "Embedding : Invalid device: " << device << std::endl;
        exit(1);
    }
    device_ = device;
}

const int BS = 512;

__global__ void embedding_forward_kernel(
    const float*  weight,
    const int*  ids,
    float*  output,
    int N,
    int K
) {
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;
    if (bx * BS + tx < K) {
        output[by * K + bx * BS + tx] = weight[ids[by] * K + bx * BS + tx];
    }
}

void embedding_forward_cpu(
    const float*  weight,
    const int*  ids,
    float*  output,
    int N,
    int K
) {
    for (int i=0; i<N; i++) {
        std::copy(&weight[ids[i] * K], &weight[ids[i] * K + K], &output[i * K]);
    }
}

void Embedding::forward(const std::vector<int>& tokens, Tensor& output) {
    int N = tokens.size();
    for (int i=0; i<N; i++) {
        if (tokens[i] < 0 || tokens[i] >= weight_.shape_[0]) {
            std::cerr << "Embedding : Invalid token: " << tokens[i] << " ( i = " << i << " ), vocabulary size = " << weight_.shape_[0] << std::endl;
            exit(1);
        }
        ids_h[i] = tokens[i];
    }
    int K = weight_.shape_[1];

    if (device_ == "cpu" && output.device_ == "cpu") {
        embedding_forward_cpu(
            weight_.h_ptr, 
            ids_h, 
            output.h_ptr, 
            N, K
        );
    }
    else if (device_ == "cuda" && output.device_ == "cuda") {
        CHECK_CUDA(cudaMemcpy(ids_d, ids_h, N * sizeof(int), cudaMemcpyHostToDevice));
        dim3 grid((K + BS - 1) / BS, N);
        dim3 block(BS);
        embedding_forward_kernel<<<grid, block>>>(
            weight_.d_ptr,
            ids_d,
            output.d_ptr,
            N,K
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else {
        std::cerr << "Embedding : Invalid device: " << device_ << " -> " << output.device_ << std::endl;
        exit(1);
    }
}

void Embedding::forward(int token, Tensor& output) {
    if (token < 0 || token >= weight_.shape_[0]) {
        std::cerr << "Embedding : Invalid token: " << token << std::endl;
        exit(1);
    }
    int K = weight_.shape_[1];
    if (device_ == "cpu" && output.device_ == "cpu") {
        std::copy(&weight_.h_ptr[token * K], &weight_.h_ptr[token * K + K], &output.h_ptr[0]);
    }
    else if (device_ == "cuda" && output.device_ == "cuda") {
        CHECK_CUDA(cudaMemcpy(output.d_ptr, &weight_.d_ptr[token * K], K * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    else {
        std::cerr << "Embedding : Invalid device: " << device_ << " -> " << output.device_ << std::endl;
        exit(1);
    }   
}

__global__ void embedding_inverse_kernel(
    const float*  weight,
    const float*  input,
    float*  output,
    int N,
    int K
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    float dot = 0.0;
    for (int i=0; i<K; i+=BS) {
        if (i + tx < K) {
            dot += input[i + tx] * weight[bx * K + i + tx];
        }
    }
    __shared__ float s_dot[BS];
    s_dot[tx] = dot;
    __syncthreads();
    for (int s=BS/2; s>0; s>>=1) {
        if (tx < s) {
            s_dot[tx] += s_dot[tx + s];
        }
        __syncthreads();
    }
    output[bx] = s_dot[0];
}

void embedding_inverse_cpu(
    const float*  weight,
    const float*  input,
    float*  output,
    int N,
    int K
) {
    # pragma omp parallel for
    for (int i=0; i<N; i++) {
        float dot = 0.0;
        for (int j=0; j<K; j++) {
            dot += input[j] * weight[i * K + j];
        }
        output[i] = dot;
    }
}

void Embedding::inverse(const Tensor& x, Tensor& output) {
    if (x.shape_[x.dims() - 1] != weight_.shape_[1] || output.dims() != 1 || output.shape_[0] != weight_.shape_[0]) {
        std::cerr << "Embedding_inv : input shape not match: " << std::endl;
        __print_vec__(x.shape_, "x");
        __print_vec__(weight_.shape_, "weights");
        exit(1);
    }
    int K = x.shape_[x.dims() - 1];
    int N = weight_.shape_[0];
    if (device_ == "cpu" && output.device_ == "cpu") {
        embedding_inverse_cpu(
            weight_.h_ptr,
            x.h_ptr,
            output.h_ptr,
            N, K
        );
    }
    else if (device_ == "cuda" && output.device_ == "cuda") {
        dim3 grid(N);
        dim3 block(BS);
        embedding_inverse_kernel<<<grid, block>>>(
            weight_.d_ptr,
            x.d_ptr,
            output.d_ptr,
            N, K
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else {
        std::cerr << "Embedding_inv : Invalid device: " << device_ << " -> " << output.device_ << std::endl;
        exit(1);
    }
}
