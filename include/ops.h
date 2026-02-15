#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include "tensor.h"

class Linear {
public:
    Tensor weight_;
    Tensor bias_;
    Tensor weight_T_; // 转置权重
    std::string device_;
public:
    Linear(int in_dim, int out_dim, std::string device);
    ~Linear();
    void load(const std::string& path);
    std::string device() const;
    void to(const std::string& device);
    // forward支持对一般的tensor变换
    // forward_vec仅支持对向量(1-dim tensor)变换
    void forward(const Tensor& input, Tensor& output);
    void forward_vec(const Tensor& input, Tensor& output);
};

class RMSNorm {
public:
    Tensor gamma_;
    std::string device_;
public:
    RMSNorm(int dim, std::string device);
    ~RMSNorm();
    void load(const std::string& path);
    std::string device() const;
    void to(const std::string& device);
    // 默认原地修改
    void forward(Tensor& input);
};

class RoPE {
public:
    static void forward(Tensor& x, float base);
    static void forward_vec(Tensor& x, int pos, float base);
};

class KVcache {
public:
    Tensor K_;
    Tensor V_;
    int n;
    std::string device_;
public:
    KVcache(int max_len, int dim, int heads, const std::string& device);
    ~KVcache();
    void to(const std::string& device);
    std::string device() const;
};

class MultiHeadAttention {
public:
    static void prefill(
        const Tensor& Q, 
        const Tensor& K, 
        const Tensor& V, 
        Tensor& output,
        KVcache& KV
    );
    static void forward(
        const Tensor& Q, 
        const Tensor& K, 
        const Tensor& V, 
        Tensor& output,
        KVcache& KV
    );
};

class Embedding {
public:
    Tensor weight_;
    int* ids_d;
    int* ids_h;
    std::string device_;
public:
    Embedding(int vocabulary, int latent_dim, const std::string& device);
    ~Embedding();
    void load(const std::string& base_path);
    std::string device() const;
    void to(const std::string& device);
    void forward(const std::vector<int>& tokens, Tensor& output);
    void forward(int token, Tensor& output);
    void inverse(const Tensor& x, Tensor& output);
    Embedding(const Tensor& Embedding) = delete;
    Embedding& operator=(const Tensor& Embedding) = delete;
};

class AttentionBlock {
public:
    Linear Q_;
    Linear K_;
    Linear V_;
    Linear W_;
    Tensor workplace_Q;
    Tensor workplace_K;
    Tensor workplace_V;
    Tensor workplace_W;
    Tensor vec_Q;
    Tensor vec_K;
    Tensor vec_V;
    Tensor vec_W;
    RoPE rope_;
    MultiHeadAttention ma_;
    KVcache KV;
    int max_len;
    int latent_dim;
    int m;
    std::string device_;
    float base_rope;
public:
    AttentionBlock(int max_len, int latent_dim, int m, float base_rope, std::string device);
    ~AttentionBlock();
    void load(const std::string& base_path);
    std::string device() const;
    void to(const std::string& device);
    void forward(const Tensor& x, Tensor& output, bool is_prefill);
};

class SwishGLU {
public:
    Linear W1_;
    Linear W2_;
    Linear V_;
    Tensor workplace_W;
    Tensor workplace_V;
    Tensor vec_W;
    Tensor vec_V;
    int hidden_dim;
    std::string device_;
public:
    SwishGLU(int in_dim, int hidden_dim, std::string device);
    ~SwishGLU();
    void load(const std::string& base_path);
    std::string device() const;
    void to(const std::string& device);
    void forward(Tensor& x);
    void forward_vec(Tensor& x);
};

class TransformerBlock {
public:
    AttentionBlock attn_;
    SwishGLU ffn_;
    RMSNorm norm1_;
    RMSNorm norm2_;
    Tensor residual_;
    std::string device_;
public:
    TransformerBlock(
        int max_len, 
        int latent_dim, 
        int m, 
        float base_rope, 
        int hidden_dim, 
        std::string device
    );
    ~TransformerBlock();
    void load(const std::string& base_path);
    void to(const std::string& device);
    std::string device() const;
    void forward(Tensor& x, bool is_prefill);

};

class Transformer {
public:
    std::vector<std::unique_ptr<TransformerBlock> > blocks_;
    Embedding embedding_;
    RMSNorm norm_;
    std::string device_;
    int max_len;
    int latent_dim;
    Tensor workplace_prefill;
    Tensor workplace_decode;
public:
    Transformer(
        int max_len, 
        int latent_dim, 
        int num_heads, 
        float base_rope, 
        int hidden_dim, 
        int vocabulary_size, 
        int num_layers,
        std::string device
    );
    ~Transformer();
    void load(const std::string& base_path);
    void to(const std::string& device);
    std::string device() const;
    void forward(const std::vector<int>& tokens, Tensor& prob_logits);
    void forward(int token, Tensor& prob_logits);
};