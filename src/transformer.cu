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

TransformerBlock::TransformerBlock(
    int max_len, 
    int latent_dim, 
    int m, 
    float base_rope, 
    int hidden_dim, 
    std::string device
) : 
    attn_(max_len, latent_dim, m, base_rope, device), 
    ffn_(latent_dim, hidden_dim, device),
    norm1_(latent_dim, device),
    norm2_(latent_dim, device),
    residual_({max_len, latent_dim}, device),
    device_(device){
}

TransformerBlock::~TransformerBlock() {
}

void TransformerBlock::load(const std::string& base_path) {
    attn_.load(base_path + "/attn");
    ffn_.load(base_path + "/ffn");
    norm1_.load(base_path + "/norm1");
    norm2_.load(base_path + "/norm2");
}

void TransformerBlock::to(const std::string& device) {
    attn_.to(device);
    ffn_.to(device);
    norm1_.to(device);
    norm2_.to(device);
    residual_.to(device);
    device_ = device;
}

std::string TransformerBlock::device() const {
    return device_;
}

const int BS = 256;
// A += B
__global__ void add_kernel(float* A, const float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        A[idx] += B[idx];
    }
}

void add_cpu(float* A, const float* B, int N) {
    # pragma omp parallel for schedule(static)
    for (int i=0; i<N; i++) {
        A[i] += B[i];
    }
}

void TransformerBlock::forward(Tensor& x, bool is_prefill) {
    if (device_ == "cpu") {
        std::copy(x.h_ptr, x.h_ptr + x.size(), residual_.h_ptr);
    }
    else {
        CHECK_CUDA(cudaMemcpy(residual_.d_ptr, x.d_ptr, x.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    residual_.shape_ = x.shape_;
    norm1_.forward(x);
    attn_.forward(x, x, is_prefill);
    if (device_ == "cpu") {
        add_cpu(x.h_ptr, residual_.h_ptr, x.size());
        std::copy(x.h_ptr, x.h_ptr + x.size(), residual_.h_ptr);
    }
    else {
        dim3 grid((x.size() + BS - 1) / BS);
        dim3 block(BS);
        add_kernel<<<grid, block>>>(x.d_ptr, residual_.d_ptr, x.size());
        CHECK_KERNEL_LAUNCH();
        CHECK_CUDA(cudaMemcpy(residual_.d_ptr, x.d_ptr, x.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    norm2_.forward(x);
    if (is_prefill){
        ffn_.forward(x);
    }
    else {
        ffn_.forward_vec(x);
    }
    if (device_ == "cpu") {
        add_cpu(x.h_ptr, residual_.h_ptr, x.size());
    }
    else {
        dim3 grid((x.size() + BS - 1) / BS);
        dim3 block(BS);
        add_kernel<<<grid, block>>>(x.d_ptr, residual_.d_ptr, x.size());
        CHECK_KERNEL_LAUNCH();
    }
}

Transformer::Transformer(
    int max_len, 
    int latent_dim, 
    int num_heads, 
    float base_rope, 
    int hidden_dim, 
    int vocabulary_size, 
    int num_layers,
    std::string device
) : 
    embedding_(vocabulary_size, latent_dim, device), 
    norm_(latent_dim, device),
    device_(device),
    max_len(max_len),
    latent_dim(latent_dim),
    workplace_prefill({max_len, latent_dim}, device),
    workplace_decode({latent_dim}, device) {
    for (int i=0; i<num_layers; i++) {
        blocks_.push_back(std::make_unique<TransformerBlock>(
            max_len, 
            latent_dim, 
            num_heads, 
            base_rope, 
            hidden_dim, 
            device
        ));
    }
}

Transformer::~Transformer() {
}

void Transformer::load(const std::string& base_path) {
    embedding_.load(base_path + "/embedding");
    norm_.load(base_path + "/norm");
    for (int i=0; i<blocks_.size(); i++) {
        blocks_[i]->load(base_path + "/blocks/" + std::to_string(i));
    }

}

void Transformer::to(const std::string& device) {
    embedding_.to(device);
    norm_.to(device);
    for (int i=0; i<blocks_.size(); i++) {
        blocks_[i]->to(device);
    }
    device_ = device;
    workplace_prefill.to(device);
    workplace_decode.to(device);
}

std::string Transformer::device() const {
    return device_;
}

void Transformer::forward(const std::vector<int>& tokens, Tensor& prob_logits) {
    int N = tokens.size(), K = latent_dim;
    workplace_prefill.shape_[0] = N;
    embedding_.forward(tokens, workplace_prefill);
    for (auto& block : blocks_) {
        block->forward(workplace_prefill, true);
    }
    norm_.forward(workplace_prefill);
    if (device_ == "cpu") {
        std::copy(workplace_prefill.h_ptr + K * (N - 1), workplace_prefill.h_ptr + K * N, workplace_decode.h_ptr);
    }
    else {
        CHECK_CUDA(cudaMemcpy(workplace_decode.d_ptr, workplace_prefill.d_ptr + K * (N - 1), K * sizeof(float), cudaMemcpyDeviceToDevice));
    }   
    embedding_.inverse(workplace_decode, prob_logits);
}

void Transformer::forward(int token, Tensor& prob_logits) {
    embedding_.forward(token, workplace_decode);
    for (auto& block : blocks_) {
        block->forward(workplace_decode, false);
    }
    norm_.forward(workplace_decode);
    embedding_.inverse(workplace_decode, prob_logits);
}