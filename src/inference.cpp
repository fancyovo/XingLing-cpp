#include "ops.h"
#include "tensor.h"
#include "inference.h"
#include <iostream>
#include <cmath>
#include <string>
#include <memory>
#include <cassert>
#include <immintrin.h>
#include <cstring>
#include <random>

InferenceEngine::InferenceEngine() : model_(
    768,        // max_seq_len
    1536,       // latent_dim
    24,         // num_heads
    10000.0,    // RoPE_base
    4096,       // ffn_dim
    151669,     // vocabulary_size
    16          // num_layers
){
    im_start_id = 151644;
    im_end_id = 151645;
    nl_id = 198;
    role_id_user = 872;
    role_id_assistant = 77091;
    selected_ids.resize(151669);
    std::random_device rd;
    gen_ = std::mt19937(rd());
}


InferenceEngine::~InferenceEngine(){
    // miao~~
}

void InferenceEngine::load(const std::string& model_path){
    model_.load(model_path);
}


int InferenceEngine::sample(float* logits, int n, float temperature, float top_p, float repetition_penalty) {
    for (auto x: repetition_set) {
        if (logits[x] > 0) logits[x] /= repetition_penalty;
        else logits[x] *= repetition_penalty;
    }
    float max_val = -1e6, sum_exp = 0.0;
    for (int i=0; i<n; i++) {
        logits[i] /= temperature + 0.01;
        pq.push({logits[i], i});
        if (logits[i] > max_val) {
            sum_exp *= std::exp(max_val - logits[i]);
            sum_exp ++;
            max_val = logits[i];
        }
        else {
            sum_exp += std::exp(logits[i] - max_val);
        }
    }
    float actural_total_prob = 0.0;
    int total_selected = 0;
    while (!pq.empty() && actural_total_prob < top_p) {
        auto x = pq.top(); pq.pop();
        float prob = std::exp(x.first - max_val) / sum_exp;
        actural_total_prob += prob;
        selected_ids[total_selected] = x.second;
        total_selected ++;
    } 
    while (!pq.empty()) pq.pop();
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    float r = dis(gen_);
    float actural_prob = 0.0;
    for (int i=0; i<total_selected; i++) {
        actural_prob += std::exp(logits[selected_ids[i]] - max_val) / sum_exp;
        if (actural_prob >= r * actural_total_prob) {
            return selected_ids[i];
        }
    }
    return selected_ids[total_selected-1];
}

int InferenceEngine::next_token(const std::vector<int>& input_ids, float temperature = 0.7, float top_p = 0.9, float repetition_penalty = 1.1){
    Tensor logits = model_.forward(input_ids);
    int n = logits.shape_[0];
    int vocab = logits.shape_[1];
    int output_id = sample(&logits.data_[vocab * (n-1)], vocab, temperature, top_p, repetition_penalty);
    repetition_set.clear();
    repetition_set.insert(output_id);
    return output_id;
}

int InferenceEngine::next_token(int token, float temperature = 0.7, float top_p = 0.9, float repetition_penalty = 1.1){
    Tensor logits = model_.forward(token);
    int output_id = sample(&logits.data_[0], logits.shape_[0], temperature, top_p, repetition_penalty);
    repetition_set.insert(output_id);
    return output_id;
}

float InferenceEngine::nll(const std::vector<int>& input_ids, const std::vector<int>& target_ids){
    Tensor logits = model_.forward(input_ids);
    int n = logits.shape_[0];
    int vocab = logits.shape_[1];
    float total_logits = 0.0;
    # pragma omp parallel for reduction(+:total_logits)
    for (int i=0; i<n; i++) {
        float max_val = -1e6, sum_exp = 0.0;
        for (int j=0; j<vocab; j++) {
            if (logits.data_[vocab * i + j] > max_val) {
                sum_exp *= std::exp(max_val - logits.data_[vocab * i + j]);
                sum_exp += 1.0;
                max_val = logits.data_[vocab * i + j];
            }
            else {
                sum_exp += std::exp(logits.data_[vocab * i + j] - max_val);
            }
        }
        total_logits += logits.data_[vocab * i + target_ids[i]] - std::log(sum_exp) - max_val;
    }
    return -total_logits / n;
}