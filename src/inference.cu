#include "ops.h"
#include "tensor.h"
#include "utils.h"
#include "inference.h"
#include <iostream>
#include <cmath>
#include <string>
#include <memory>
#include <cassert>
#include <immintrin.h>
#include <cstring>
#include <random>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>


InferenceEngine::InferenceEngine(std::string device) : 
    model_(
        768,        // max_seq_len
        1536,       // latent_dim
        24,         // num_heads
        10000.0,    // RoPE_base
        4096,       // ffn_dim
        151669,     // vocabulary_size
        16,         // num_layers
        device
    ),
    logits({151669}, device),
    device_(device)
{
    im_start_id = 151644;
    im_end_id = 151645;
    nl_id = 198;
    role_id_user = 872;
    role_id_assistant = 77091;
    std::random_device rd;
    gen_ = std::mt19937(rd());
}

InferenceEngine::~InferenceEngine(){
}

void InferenceEngine::load(const std::string& model_path){
    model_.load(model_path);
}

void InferenceEngine::to(const std::string& device){
    model_.to(device);
    logits.to(device);
    device_ = device;
}

std::string InferenceEngine::device() const {
    return device_;
}

struct LogitsIdx {
    int idx;
    float val;
    bool operator<(const LogitsIdx& other) const {
        return val > other.val;
    }
}P[151669];

// 我们认为前1024个token已经覆盖TopP的值了。
const int top_k = 1024;

int sample_cpu(
    float* logits, 
    int n, 
    float temperature, 
    float top_p, 
    float repetition_penalty,
    std::mt19937& gen,
    std::set<int>& repetition_set
) {
    for (auto x: repetition_set) {
        if (logits[x] > 0) logits[x] /= repetition_penalty;
        else logits[x] *= repetition_penalty;
    }
    for (int i=0; i<n; i++) {
        P[i] = {i, logits[i]};
    }
    std::nth_element(P, P + top_k, P + n);
    std::sort(P, P + top_k);
    float max_val = -1e6f, sum_exp = 0.0f;
    for (int i=0; i<top_k; i++) {
        float val = P[i].val / (temperature + 0.001f);
        P[i].val = val;
        if (val > max_val) {
            sum_exp = sum_exp * std::exp(max_val - val) + 1.0f;
            max_val = val;
        }
        else {
            sum_exp += std::exp(val - max_val);
        }
    }
    int max_pos;
    float total_prob = 0.0f;
    for (max_pos=0; max_pos<top_k; max_pos++) {
        total_prob += std::exp(P[max_pos].val - max_val) / sum_exp;
        if (total_prob >= top_p) break;
    }
    float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(gen);
    float sum_prob = 0.0f;
    for (int i=0; i<top_k; i++) {
        sum_prob += std::exp(P[i].val - max_val) / sum_exp;
        if (sum_prob > r * total_prob) {
            return P[i].idx;
        }
    }
    return P[top_k - 1].idx;
}

int InferenceEngine::next_token(const std::vector<int>& input_ids, float temperature, float top_p, float repetition_penalty) {
    if (input_ids.size() > model_.max_len) {
        std::cerr << "Input sequence length exceeds max_seq_len" << std::endl;
        exit(1);
    }
    repetition_set.clear();
    model_.forward(input_ids, logits);
    if (device_ == "cpu") {
        int token = sample_cpu(logits.h_ptr, logits.size(), temperature, top_p, repetition_penalty, gen_, repetition_set);
        repetition_set.insert(token);
        return token;
    }
    else {
        logits.to("cpu");
        int token = sample_cpu(logits.h_ptr, logits.size(), temperature, top_p, repetition_penalty, gen_, repetition_set);
        repetition_set.insert(token);
        logits.device_ = device_;
        return token;
        
    }
}

int InferenceEngine::next_token(int input_id, float temperature, float top_p, float repetition_penalty) {
    model_.forward(input_id, logits);
    if (device_ == "cpu") {
        int token = sample_cpu(logits.h_ptr, logits.size(), temperature, top_p, repetition_penalty, gen_, repetition_set);
        repetition_set.insert(token);
        return token;
    }
    else {
        logits.to("cpu");
        int token = sample_cpu(logits.h_ptr, logits.size(), temperature, top_p, repetition_penalty, gen_, repetition_set);
        repetition_set.insert(token);
        logits.device_ = device_;
        return token;
    }
}