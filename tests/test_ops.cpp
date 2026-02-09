#include "ops.h"
#include "tensor.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#include <functional>
#include <iomanip>
#include <omp.h>
#include <cassert>
#include <cstring>
#include <algorithm>

using namespace std;

// ==========================================
// 全局配置与工具
// ==========================================

// 0.7B 模型参数配置
const int CFG_SEQ_LEN = 768;
const int CFG_LATENT = 1536;
const int CFG_FFN = 4096;
const int CFG_HEADS = 24;
const int CFG_LAYERS = 16;
const int CFG_HEAD_DIM = CFG_LATENT / CFG_HEADS; // 64
const float CFG_ROPE_BASE = 10000.0f;
const int CFG_VOCAB = 151669; // 假设词表大小

// 随机数生成器
void rand_tensor(Tensor& t, float min = -0.1f, float max = 0.1f) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min, max);
    int size = t.size();
    // 使用指针访问避免 operator[] 的开销，加快初始化
    float* ptr = t.data_.get();
    for (int i = 0; i < size; ++i) {
        ptr[i] = dis(gen);
    }
}

// 相对误差比较
bool check_close(const Tensor& a, const Tensor& b, float rtol = 1e-2, float atol = 1e-3) {
    if (a.shape_ != b.shape_) {
        std::cerr << "[ERROR] Shape mismatch: " << a.shape_[0] << "... vs " << b.shape_[0] << "..." << std::endl;
        return false;
    }
    int size = a.size();
    float max_diff = 0.0f;
    float max_val_a = 0.0f;
    float max_val_b = 0.0f;
    
    for(int i=0; i<size; ++i) {
        float v1 = a.data_[i];
        float v2 = b.data_[i];
        float diff = std::abs(v1 - v2);
        if (diff > atol + rtol * std::abs(v2)) {
            // 打印第一个错误
            std::cerr << "[ERROR] Mismatch at index " << i << ": " << v1 << " vs " << v2 << std::endl;
            return false;
        }
        max_diff = std::max(max_diff, diff);
        max_val_a = std::max(max_val_a, std::abs(v1));
    }
    return true;
}

// 稳健的 Benchmark 函数
// min_duration_sec: 最小运行秒数，防止运行太快测不准
void benchmark(const std::string& name, double flops_per_iter, std::function<void()> func, double min_duration_sec = 1.0) {
    // 1. Warmup (唤醒 CPU，预热 Cache)
    for(int i=0; i<3; ++i) func();

    // 2. 动态决定迭代次数
    auto t1 = std::chrono::high_resolution_clock::now();
    func();
    auto t2 = std::chrono::high_resolution_clock::now();
    double single_run_time = std::chrono::duration<double>(t2 - t1).count();
    
    int iters = std::max(5, int(min_duration_sec / (single_run_time + 1e-9)));
    
    // 3. 正式测试
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iters; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_time = std::chrono::duration<double>(end - start).count();
    double avg_time = total_time / iters;
    double gflops = (flops_per_iter / avg_time) / 1e9;

    std::cout << std::left << std::setw(30) << name 
              << "| GFLOPS: " << std::setw(8) << std::fixed << std::setprecision(2) << gflops
              << "| Latency: " << std::setw(8) << std::setprecision(4) << avg_time * 1000.0 << " ms" 
              << std::endl;
}

// ==========================================
// 1. 基础算子测试 (Functions)
// ==========================================
void test_functions() {
    std::cout << "\n[1. Testing Basic Functions]" << std::endl;
    int size = CFG_LATENT * 1024; // 1.5M elements
    Tensor x({size});
    rand_tensor(x, 0.1f, 2.0f); // 避免 log/sqrt 负数
    
    // 简单正确性
    Tensor y = Functions::silu(x);
    if (std::abs(y.data_[0] - (x.data_[0] / (1.0f + std::exp(-x.data_[0])))) > 1e-4) {
        std::cerr << "SiLU Correctness Check Failed!" << std::endl;
    }

    // 测速 (FLOPs 估算: SiLU ~ 10 ops, Exp ~ 10 ops)
    benchmark("Functions::SiLU", size * 10.0, [&](){ Functions::silu(x); });
    benchmark("Functions::Exp", size * 10.0, [&](){ Functions::Exp(x); });
    
    // Softmax (Last dim = 4096)
    Tensor sm_in({256, CFG_FFN}); 
    rand_tensor(sm_in);
    benchmark("Functions::Softmax", 256.0 * CFG_FFN * 4.0, [&](){ Functions::softmax(sm_in); });
    
    // Matmul: [B, I] * [O, I]^T -> [B, O]
    // 模拟 FFN: [128, 1536] * [4096, 1536]^T
    // ops.cpp matmul 实现是 (Out, In) * (Batch, In)^T -> (Batch, Out)
    // 即 A是权重，B是输入
    int B = 128;
    int I = CFG_LATENT; // 1536
    int O = CFG_FFN;    // 4096
    Tensor W({O, I}); rand_tensor(W);
    Tensor X({B, I}); rand_tensor(X);
    
    // FLOPs = 2 * M * N * K = 2 * B * O * I
    benchmark("Matmul [128, 1536]x[1536, 4096]", 2.0 * B * O * I, [&](){ Functions::matmul(W, X); });
}

// ==========================================
// 2. 层级组件测试 (Layers)
// ==========================================
void test_layers() {
    std::cout << "\n[2. Testing Layers]" << std::endl;
    
    // --- Linear ---
    int batch = 128;
    Linear lin(CFG_LATENT, CFG_LATENT); // Projection layer
    rand_tensor(lin.coef_); rand_tensor(lin.bias_);
    Tensor x({batch, CFG_LATENT}); rand_tensor(x);
    
    benchmark("Linear [1536->1536]", 2.0 * batch * CFG_LATENT * CFG_LATENT, [&](){ lin.forward(x); });

    // --- RMSNorm ---
    RMSNorm norm(CFG_LATENT);
    rand_tensor(norm.gamma_);
    benchmark("RMSNorm", 4.0 * batch * CFG_LATENT, [&](){ norm.forward(x); });

    // --- RoPE ---
    // 验证正确性：旋转后模长不变
    RoPE rope(CFG_ROPE_BASE, 2048, CFG_HEAD_DIM);
    Tensor q({1, 10, CFG_HEADS, CFG_HEAD_DIM}); // [1, seq, n, d]
    rand_tensor(q);
    Tensor q_orig = q.clone();
    rope.forward(q);
    
    float norm_orig = 0, norm_rope = 0;
    for(int i=0; i<CFG_HEAD_DIM; ++i) { // Check first head first token
        norm_orig += q_orig.data_[i] * q_orig.data_[i];
        norm_rope += q.data_[i] * q.data_[i];
    }
    if (std::abs(norm_orig - norm_rope) > 1e-3) std::cerr << "RoPE Norm Check Failed!" << std::endl;
    
    benchmark("RoPE (Seq=128)", 128.0 * CFG_HEADS * CFG_HEAD_DIM * 4.0, [&](){
        Tensor temp({1, 128, CFG_HEADS, CFG_HEAD_DIM});
        rope.forward(temp); 
    });

    // --- SwishGLU ---
    // 输入: [Batch, Latent], 内部: Up(Latent->FFN), Gate(Latent->FFN), Down(FFN->Latent)
    SwishGLU glu(CFG_LATENT, CFG_FFN);
    rand_tensor(glu.W1_.coef_); rand_tensor(glu.W2_.coef_); rand_tensor(glu.V_.coef_);
    rand_tensor(glu.W1_.bias_); rand_tensor(glu.W2_.bias_); rand_tensor(glu.V_.bias_);
    
    // FLOPs: 3 * Matmul
    double glu_flops = 2.0 * batch * CFG_LATENT * CFG_FFN * 3; 
    benchmark("SwishGLU", glu_flops, [&](){ glu.forward(x); });
}

// ==========================================
// 3. 注意力模块测试 (Attention)
// ==========================================

// 朴素 Attention 实现，用于验证正确性
Tensor naive_attn(const Tensor& Q, const Tensor& K, const Tensor& V) {
    // Q,K,V: [B, Seq, Heads, Dim]
    int B = Q.shape_[0]; int N = Q.shape_[1]; int H = Q.shape_[2]; int D = Q.shape_[3];
    Tensor out(Q.shape_);
    float scale = 1.0f / std::sqrt((float)D);
    
    for(int b=0; b<B; ++b) {
        for(int h=0; h<H; ++h) {
            for(int i=0; i<N; ++i) { // Q idx
                std::vector<float> scores;
                float max_val = -1e9;
                for(int j=0; j<=i; ++j) { // K idx (Causal)
                    float dot = 0;
                    for(int d=0; d<D; ++d) {
                        dot += Q.data_[b*N*H*D + i*H*D + h*D + d] * K.data_[b*N*H*D + j*H*D + h*D + d];
                    }
                    dot *= scale;
                    scores.push_back(dot);
                    max_val = std::max(max_val, dot);
                }
                float sum_exp = 0;
                for(auto& s : scores) { s = std::exp(s - max_val); sum_exp += s; }
                
                for(int d=0; d<D; ++d) {
                    float val = 0;
                    for(int j=0; j<=i; ++j) {
                        val += (scores[j] / sum_exp) * V.data_[b*N*H*D + j*H*D + h*D + d];
                    }
                    out.data_[b*N*H*D + i*H*D + h*D + d] = val;
                }
            }
        }
    }
    return out;
}

void test_attention() {
    std::cout << "\n[3. Testing Attention (0.7B Config)]" << std::endl;
    int B = 1; 
    int Seq = 32; // 短序列验证正确性
    Tensor Q({B, Seq, CFG_HEADS, CFG_HEAD_DIM});
    Tensor K({B, Seq, CFG_HEADS, CFG_HEAD_DIM});
    Tensor V({B, Seq, CFG_HEADS, CFG_HEAD_DIM});
    rand_tensor(Q); rand_tensor(K); rand_tensor(V);
    
    // 1. Correctness: Prefill vs Naive
    KVcache kv(128, CFG_HEAD_DIM, CFG_HEADS);
    Tensor out_opt = MultiHeadAttention::prefill(Q, K, V, kv);
    Tensor out_ref = naive_attn(Q, K, V);
    
    if (check_close(out_opt, out_ref, 1e-3)) {
        std::cout << "Attention Logic Check: PASS" << std::endl;
    } else {
        std::cerr << "Attention Logic Check: FAIL" << std::endl;
    }
    
    // 2. Benchmark Prefill (Seq=512)
    int seq_pf = 512;
    Tensor Q_pf({B, seq_pf, CFG_HEADS, CFG_HEAD_DIM});
    Tensor K_pf({B, seq_pf, CFG_HEADS, CFG_HEAD_DIM});
    Tensor V_pf({B, seq_pf, CFG_HEADS, CFG_HEAD_DIM});
    rand_tensor(Q_pf); rand_tensor(K_pf); rand_tensor(V_pf);
    
    // FLOPs: 4 * B * H * N^2 * D
    double pf_flops = 4.0 * B * CFG_HEADS * std::pow(seq_pf, 2) * CFG_HEAD_DIM;
    benchmark("Attn Prefill (Seq=512)", pf_flops, [&](){
        KVcache temp_kv(seq_pf, CFG_HEAD_DIM, CFG_HEADS);
        MultiHeadAttention::prefill(Q_pf, K_pf, V_pf, temp_kv);
    });
    
    // 3. Benchmark Decode (Context=512)
    // 模拟已经有了 512 的 KV Cache，计算下一个 token
    KVcache kv_dec(1024, CFG_HEAD_DIM, CFG_HEADS);
    MultiHeadAttention::prefill(Q_pf, K_pf, V_pf, kv_dec); // 填充 Cache
    
    Tensor q_step({CFG_HEADS, CFG_HEAD_DIM});
    Tensor k_step({CFG_HEADS, CFG_HEAD_DIM});
    Tensor v_step({CFG_HEADS, CFG_HEAD_DIM});
    rand_tensor(q_step); rand_tensor(k_step); rand_tensor(v_step);
    
    // FLOPs: 4 * H * Context * D
    double dec_flops = 4.0 * CFG_HEADS * seq_pf * CFG_HEAD_DIM;
    benchmark("Attn Decode (Ctx=512)", dec_flops, [&](){
        // 注意：每次调用 forward kv.n 会 +1，长期运行可能会越界
        // 但 benchmark 内部运行时间不长，1024 cache 足够支撑几千次迭代
        // 且为了性能测试准确性，不应在循环内重置 cache
        MultiHeadAttention::forward(q_step, k_step, v_step, kv_dec);
        kv_dec.n = 513; // 重置 cache 指针
    });
}

// ==========================================
// 4. 模型整体性能测试 (TTFT & Decode Speed)
// ==========================================
void test_model() {
    std::cout << "\n[4. Testing Model Performance]" << std::endl;
    setenv("OMP_WAIT_POLICY", "active", 1);
    
    // --- 测试配置 ---
    const int PROMPT_LEN = 128; // Prompt 长度
    const int GEN_LEN = 100;    // 生成 Token 数量

    // 1. 加载模型
    Transformer model(CFG_SEQ_LEN, CFG_LATENT, CFG_HEADS, CFG_ROPE_BASE, CFG_FFN, CFG_VOCAB, CFG_LAYERS);
    std::string path = "../data/model";
    
    try {
        auto t_start = std::chrono::high_resolution_clock::now();
        model.load(path);
        auto t_end = std::chrono::high_resolution_clock::now();
        std::cout << "Model loaded in " << std::chrono::duration<double>(t_end - t_start).count() << " s." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FATAL] Model load failed: " << e.what() << std::endl;
        return;
    }

    // 2. 构造随机 Prompt
    std::cout << "Config: Prompt=" << PROMPT_LEN << " tokens, Generate=" << GEN_LEN << " tokens" << std::endl;
    std::vector<int> prompt(PROMPT_LEN);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, CFG_VOCAB - 1);
    for(int i=0; i<PROMPT_LEN; ++i) prompt[i] = dis(gen);

    // 3. Warmup (重要：激活内存分配和线程池)
    std::cout << "Warming up..." << std::endl;
    {
        std::vector<int> dummy = {1}; 
        model.forward(dummy); // 触发一次微型 Prefill
        model.forward(1);     // 触发一次微型 Decode
    }

    std::cout << "------------------------------------------------" << std::endl;

    // --- Phase 1: Prefill (首字延迟 TTFT) ---
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Forward pass: 处理整个 Prompt
    Tensor logits = model.forward(prompt);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double ttft_ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0;
    
    // 简单的 Argmax 采样获取下一个 token
    // logits shape: [seq, vocab], 我们取最后一个时间步
    int next_token = 0;
    {
        int vocab_size = CFG_VOCAB;
        int seq_len = PROMPT_LEN;
        float max_val = -1e9;
        int offset = (seq_len - 1) * vocab_size;
        float* ptr = logits.data_.get() + offset;
        
        // 寻找最大值 (Argmax)
        for (int i = 0; i < vocab_size; ++i) {
            if (ptr[i] > max_val) {
                max_val = ptr[i];
                next_token = i;
            }
        }
    }

    std::cout << "First Token Latency (TTFT) : " << std::fixed << std::setprecision(2) << ttft_ms << " ms" << std::endl;

    // --- Phase 2: Decode (推理速度) ---
    auto t2 = std::chrono::high_resolution_clock::now();
    
    // 自回归生成循环
    for (int i = 0; i < GEN_LEN; ++i) {
        // 单步推理
        Tensor out = model.forward(next_token);
        
        // Argmax 采样
        // out shape: [vocab] (1D)
        float max_val = -1e9;
        int best_idx = 0;
        float* ptr = out.data_.get();
        for (int v = 0; v < CFG_VOCAB; ++v) {
            if (ptr[v] > max_val) {
                max_val = ptr[v];
                best_idx = v;
            }
        }
        next_token = best_idx;
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    
    // 统计结果
    double total_gen_time = std::chrono::duration<double>(t3 - t2).count();
    double tps = GEN_LEN / total_gen_time;
    double ms_per_token = total_gen_time * 1000.0 / GEN_LEN;

    std::cout << "Decode Speed               : " << std::setprecision(2) << tps << " tokens/s" << std::endl;
    std::cout << "Avg Latency per Token      : " << std::setprecision(2) << ms_per_token << " ms" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}


int main() {
    // 设置 OpenMP 线程数，防止波动
    // 如果没有环境变量，默认使用物理核心数
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    
    std::cout << "============================================" << std::endl;
    std::cout << "      LLM Ops Benchmark (CPP)               " << std::endl;
    std::cout << "      Threads: " << max_threads << std::endl;
    std::cout << "============================================" << std::endl;

    test_functions();
    test_layers();
    test_attention();
    test_model();

    return 0;
}
