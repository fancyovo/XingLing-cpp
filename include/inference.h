#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include "tensor.h"
#include <cmath>
#include "ops.h"
#include <set>
#include <queue>
#include <utility>
#include <random>

class InferenceEngine{
private:
    Transformer model_;
    std::mt19937 gen_;
    std::set<int> repetition_set;
    int sample(float* logits, int n, float temperature, float top_p, float repetition_penalty);
    std::priority_queue<std::pair<float, int> > pq;
    std::vector<int> selected_ids;
public:
    int im_start_id;
    int im_end_id;
    int nl_id;
    int role_id_user;
    int role_id_assistant;
public:
    InferenceEngine();
    ~InferenceEngine();
    void load(const std::string& model_path);
    int next_token(const std::vector<int>& input_ids, float temperature, float top_p, float repetition_penalty);
    int next_token(int input_id, float temperature, float top_p, float repetition_penalty);
    float nll(const std::vector<int>& input_ids, const std::vector<int>& target_ids);
};