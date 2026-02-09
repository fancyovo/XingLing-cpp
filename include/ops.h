#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include "tensor.h"
#include <cmath>

class Functions {
public:
    // 初等函数
    static Tensor Exp(const Tensor& x);
    static Tensor Log(const Tensor& x);
    static Tensor Sin(const Tensor& x);
    static Tensor Cos(const Tensor& x);
    static Tensor Sqrt(const Tensor& x);
    static Tensor Pow(const Tensor& x, float p);

    // 激活函数
    static Tensor sigmoid(const Tensor& x);
    static Tensor tanh(const Tensor& x);
    static Tensor relu(const Tensor& x);
    static Tensor silu(const Tensor& x);
    static Tensor softmax(const Tensor& x);

    // 矩阵乘法、向量点积
    // 这里只考虑左矩阵右向量，即(a,b) * (...,b) -> (...,a)
    static Tensor matmul(const Tensor& A, const Tensor& B);
    static void matmul_inplace(const Tensor& A, const Tensor& B, Tensor& res);
    static Tensor dot(const Tensor& A, const Tensor& B);
};

class Linear {
public:
    Tensor coef_;
    Tensor bias_;

public:
    Linear(int in_dim, int out_dim);
    ~Linear();
    void load(const std::string& base_path);
    Tensor forward(const Tensor& x) const;
    void forward_inplace(const Tensor& x, Tensor& res) const;
};

// 默认对最后一维归一化
class RMSNorm {
public:
    Tensor gamma_;

public:
    RMSNorm(int dim);
    ~RMSNorm();
    void load(const std::string& base_path);
    Tensor forward(const Tensor& x) const;
};


// 约定对每个注意力头分别作RoPE
// 即 (B, n, m, C/m)  对 C/m 这一维作RoPE
// 推理阶段，需要考虑单token的编码，此时要传入位置参数
class RoPE {
public:
    float base_;
    int max_len_;
    int dim_;
    Tensor P;

public:
    RoPE(float base, int max_len, int dim);
    ~RoPE();
    void forward(Tensor& x) const;
    void forward(Tensor& x, int pos) const;

};

class SwishGLU {
public:
    Linear W1_;
    Linear W2_;
    Linear V_;

public:
    SwishGLU(int in_dim, int hidden_dim);
    ~SwishGLU();
    void load(const std::string& base_path);
    Tensor forward(const Tensor& x) const;
};

// KV cache 中，默认形状为 (max_len, m, C/m)
class KVcache {
public:
    Tensor K_;
    Tensor V_;
    int n;
public:
    KVcache(int max_len, int dim, int heads);
    ~KVcache();

};


// 只做多头注意力这一步，即Q,K,V均是(B, n, m, C/m)
// 默认使用因果掩码
// prefill 功能是 (B, n, m, C/m) -> (B, n, m, C/m)
// forward 只算当前向量，即 (m, C/m) -> (m, C/m)
class MultiHeadAttention {
public:
    static Tensor prefill(const Tensor& Q, const Tensor& K, 
                            const Tensor& V, KVcache& KV);
    static Tensor forward(const Tensor& Q, const Tensor& K, 
                            const Tensor& V, KVcache& KV);
};


class Embedding {
public:
    Tensor weight_;
    Tensor workplace_;
    int vocabulary;
    int latent_dim;
public:
    Embedding(int vocabulary, int latent_dim);
    ~Embedding();
    void load(const std::string& base_path);
    Tensor forward(const std::vector<int>& tokens) const;
    Tensor forward(int token) const;
    Tensor inverse(const Tensor& x);
};

// KV cache 的信息直接存储在 AttentionBlock 里
// 默认 is_prefill = True 时清空 KV cache 并重置，否则利用原 KV cache 进行计算
class AttentionBlock {
public:
    Linear Q_, K_, V_, W_;
    RoPE rope_;
    KVcache KV;
    int max_len, latent_dim, m;
public:
    AttentionBlock(int max_len, int latent_dim, int m, float base_rope);
    ~AttentionBlock();
    void load(const std::string& base_path);
    Tensor forward(const Tensor& x, bool is_prefill);
};

class TransformerBlock {
public:
    AttentionBlock attn_;
    SwishGLU ffn_;
    RMSNorm norm1_, norm2_;
public:
    TransformerBlock(int max_len, int latent_dim, int m, float base_rope, int hidden_dim);
    ~TransformerBlock();
    void load(const std::string& base_path);
    void forward(Tensor& x, bool is_prefill);
};

class Transformer {
public:
    std::vector<TransformerBlock> blocks_;
    Embedding embedding_;
    RMSNorm norm_;

public:
    Transformer(int max_len, int latent_dim, int num_heads, float base_rope, 
        int hidden_dim, int vocabulary_size, int num_layers);
    ~Transformer();
    void load(const std::string& base_path);
    Tensor forward(const std::vector<int>& tokens);
    Tensor forward(int token);
};