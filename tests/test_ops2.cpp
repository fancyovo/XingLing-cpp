#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <cassert>
#include <omp.h>
#include "tensor.h"
#include "ops.h"

using namespace std;

// ==========================================
// 工具函数：随机化、校验、性能计时
// ==========================================

void randomize_tensor(Tensor& t, float min_val = -0.5f, float max_val = 0.5f) {
    static mt19937 gen(42);
    uniform_real_distribution<float> dis(min_val, max_val);
    int s = t.size();
    for (int i = 0; i < s; ++i) {
        t.data_[i] = dis(gen);
    }
}

// 相对误差检查
bool check_close(const Tensor& a, const Tensor& b, float rtol = 1e-3, float atol = 1e-4) {
    Tensor a_c = a.contiguous();
    Tensor b_c = b.contiguous();
    if (a_c.shape_ != b_c.shape_) {
        std::cout << "Shape mismatch: "<< std::endl;
        std::cout << " a : "; for (auto x : a_c.shape_) std::cout << x << " "; std::cout << std::endl;
        std::cout << " b : "; for (auto x : b_c.shape_) std::cout << x << " "; std::cout << std::endl;
        return false;
    }
    int s = a_c.size();
    for (int i = 0; i < s; ++i) {
        float diff = std::abs(a_c.data_[i] - b_c.data_[i]);
        float tol = atol + rtol * std::abs(b_c.data_[i]);
        if (diff > tol) {
            std::cout << "Mismatch at " << i << ": val_a=" << a_c.data_[i] 
                      << ", val_b=" << b_c.data_[i] << " diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

// 性能测试通用模板
// op_name: 算子名称
// flops_per_iter: 单次迭代的浮点运算次数
// iters: 迭代次数
// func: 运行的lambda函数
void benchmark(const std::string& op_name, double flops_per_iter, int iters, std::function<void()> func) {
    // 预热
    func(); 
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iters; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    double seconds = diff.count();
    double total_flops = flops_per_iter * iters;
    double gflops = (total_flops / seconds) / 1e9;
    
    std::cout << std::left << std::setw(20) << op_name 
              << "| Time: " << std::setw(8) << std::fixed << std::setprecision(4) << (seconds/iters)*1000 << " ms"
              << "| Perf: " << std::setw(8) << std::fixed << std::setprecision(2) << gflops << " GFLOPS" 
              << std::endl;
}

// ==========================================
// 单元测试用例
// ==========================================

void test_elementwise() {
    std::cout << "=== Testing Elementwise Ops ===" << std::endl;
    int size = 1024 * 1024; // 1M elements
    Tensor x({size});
    randomize_tensor(x, 0.1f, 2.0f); // 避免 log(负数)
    
    // 1. Correctness Check (Sampling)
    Tensor y = Functions::silu(x);
    float expected = x.data_[0] / (1.0f + std::exp(-x.data_[0]));
    assert(std::abs(y.data_[0] - expected) < 1e-5);
    std::cout << "[PASS] Silu Correctness" << std::endl;

    // 2. Benchmark
    // SiLU approx FLOPs: 1 div, 1 add, 1 exp, 1 neg (exp is expensive, count as ~10 FLOPs for rough benchmarking)
    // Bandwidth usually limits elementwise, but calculating approx GFLOPS.
    benchmark("SiLU", size * 10.0, 50, [&](){ Functions::silu(x); });
    benchmark("Exp", size * 10.0, 50, [&](){ Functions::Exp(x); });
}

void test_matmul() {
    std::cout << "=== Testing Matmul ===" << std::endl;
    // LLM 中典型的形状: [Batch*Seq, Hidden] * [Hidden, Out]^T
    // ops.cpp 实现的是 (Out, In) * (Batch, In)^T -> (Batch, Out)
    int M = 128;   // Batch * Seq
    int K = 4096;  // In Dim
    int N = 4096;  // Out Dim
    
    Tensor A({N, K}); // Weight
    Tensor B({M, K}); // Input (Activation)
    randomize_tensor(A);
    randomize_tensor(B);

    // 1. Correctness Check (Small Scale)
    {
        int m=2, k=16, n=16;
        Tensor a({n, k}); randomize_tensor(a);
        Tensor b({m, k}); randomize_tensor(b);
        Tensor c = Functions::matmul(a, b);
        
        // Naive CPU Check
        float val = 0;
        for(int p=0; p<k; ++p) val += a.data_[0*k + p] * b.data_[0*k + p]; // row 0 of a, row 0 of b
        // ops.cpp matmul result index: i + j * A_size(rows). i is row of A, j is row of B.
        // res[0 + 0 * n]
        if(std::abs(c.data_[0] - val) > 1e-3) {
            std::cout << "[FAIL] Matmul Logic Error! Expected " << val << " got " << c.data_[0] << std::endl;
        } else {
            std::cout << "[PASS] Matmul Correctness (Small)" << std::endl;
        }
    }

    // 2. Benchmark
    // FLOPs = 2 * M * N * K
    double flops = 2.0 * M * N * K;
    benchmark("Matmul [128,4k,4k]", flops, 5, [&](){ Functions::matmul(A, B); });
}

void test_rmsnorm() {
    std::cout << "=== Testing RMSNorm ===" << std::endl;
    int dim = 4096;
    int batch = 128;
    RMSNorm rms(dim);
    // Initialize gamma to 1s for checking normalization logic
    for(int i=0; i<dim; ++i) rms.gamma_.data_[i] = 1.0f; 
    
    Tensor x({batch, dim});
    randomize_tensor(x, 10.0f, 20.0f); // Large values

    // 1. Correctness
    rms.forward(x);
    // Check if squared sum average is close to 1
    float sum_sq = 0;
    for(int i=0; i<dim; ++i) sum_sq += x.data_[i] * x.data_[i];
    float mean_sq = sum_sq / dim;
    if(std::abs(mean_sq - 1.0f) < 1e-3) {
        std::cout << "[PASS] RMSNorm Correctness" << std::endl;
    } else {
        std::cout << "[FAIL] RMSNorm Result: " << mean_sq << " (Expected 1.0)" << std::endl;
    }

    // 2. Benchmark
    // FLOPs approx: batch * (dim * (mul + add) + sqrt + div + dim * mul) ~ 4 * batch * dim
    benchmark("RMSNorm", 4.0 * batch * dim, 100, [&](){ rms.forward(x); });
}

void test_softmax() {
    std::cout << "=== Testing Softmax ===" << std::endl;
    int batch = 128;
    int dim = 10000; // Vocab size scale
    Tensor x({batch, dim});
    randomize_tensor(x);

    // 1. Correctness
    Tensor y = Functions::softmax(x);
    float sum = 0;
    for(int i=0; i<dim; ++i) sum += y.data_[i];
    if(std::abs(sum - 1.0f) < 1e-4) {
        std::cout << "[PASS] Softmax Sum Check" << std::endl;
    } else {
        std::cout << "[FAIL] Softmax Sum: " << sum << std::endl;
    }

    // 2. Benchmark
    // FLOPs: Exp + Sum + Div ~ 4 * elements
    benchmark("Softmax", 4.0 * batch * dim, 50, [&](){ Functions::softmax(x); });
}

void test_rope() {
    std::cout << "=== Testing RoPE ===" << std::endl;
    // Shape: (Batch, Heads, SeqLen, HeadDim)
    // 根据 ops.cpp: 维度 dim = HeadDim
    // RoPE 作用在 shape 的倒数第三维 n (SeqLen) ? 
    // Wait, ops.cpp code says: x.shape_[x.shape_.size() - 3] is max_len related
    // Let's align with ops.cpp logic:
    // int n = x.shape_[size-3]; (SeqLen)
    // int m = x.shape_[size-2]; (Heads)
    // int dim = x.shape_[size-1]; (HeadDim)
    
    int B = 1;
    int seq_len = 64;
    int heads = 8;
    int head_dim = 64; // Must be even
    
    Tensor x({B, seq_len, heads, head_dim}); 
    // Init: Set (0, 0, 0, 0) = 1, (0, 0, 0, 1) = 0
    // Theoretically after RoPE pos 0: cos(0)=1, sin(0)=0. Result should be unchanged.
    // Let's test pos 1.
    for(int i=0; i<x.size(); ++i) x.data_[i] = 0.0f;
    
    // Set x at pos=1, head=0, dim_idx=0 to 1.0
    // Index = 1 * heads * dim + 0 + 0 = 1 * 8 * 64
    int target_idx = 1 * heads * head_dim; 
    x.data_[target_idx] = 1.0f; 
    
    RoPE rope(10000.0f, 1024, head_dim);

    // 1. Correctness
    // Copy x to verify
    Tensor y = x.clone(); // Need clone() in Tensor class
    // Manual clone if not implemented:
    Tensor z(x.shape_);
    for(int i=0; i<x.size(); ++i) z.data_[i] = x.data_[i];
    
    rope.forward(z);
    
    // Calculate expected rotation for pos 1, dim 0
    // theta = base^(-0/dim) * 1 = 1 * 1 = 1 radian
    float theta = 1.0f;
    float expected_cos = std::cos(theta);
    float expected_sin = std::sin(theta);
    // Formula: x0' = x0*cos - x1*sin (Wait, ops.cpp implementation check)
    // ops.cpp: x0_rope = x[idx] * cosP + x[idx+dim/2] * sinP
    //          x1_rope = -x[idx] * sinP + ...
    // Note: ops.cpp implements x * cos + x_half * sin. 
    // If x[idx]=1, x[idx+half]=0. -> x0' = cos. x1' = -sin.
    
    float val0 = z.data_[target_idx];
    float val1 = z.data_[target_idx + head_dim/2];
    
    bool pass = true;
    if(std::abs(val0 - expected_cos) > 1e-4) pass = false;
    // ops.cpp line: x1_rope = -x[idx_x] * sinP...
    if(std::abs(val1 - (-1.0f * expected_sin)) > 1e-4) pass = false;
    
    if(pass) std::cout << "[PASS] RoPE Rotation Logic" << std::endl;
    else std::cout << "[FAIL] RoPE: Got (" << val0 << "," << val1 << ") Expected (" << expected_cos << "," << -expected_sin << ")" << std::endl;

    // 2. Benchmark
    // RoPE is heavily memory bound, but calculating FLOPs:
    // For each pair: 4 muls + 2 adds + trig lookups (cached) -> approx 6-8 FLOPs per 2 elements.
    // Total: elements * 4
    benchmark("RoPE", 4.0 * x.size(), 100, [&](){ rope.forward(x); });
}
/**

Tensor naive_attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
    int B = Q.shape_[0];
    int N = Q.shape_[1]; // Seq
    int H = Q.shape_[2]; // Heads
    int D = Q.shape_[3]; // Dim
    
    Tensor out(Q.shape_);
    float scale = 1.0f / std::sqrt((float)D);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < N; ++i) { // Query pos
                
                // 1. 计算 Attention Scores (Q * K^T)
                std::vector<float> scores;
                float max_score = -1e9;
                
                for (int j = 0; j <= i; ++j) { // Key pos (Causal Mask: j <= i)
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        // Indexing: [b, n, h, d]
                        // stride_n = H * D, stride_h = D
                        int idx_q = b*N*H*D + i*H*D + h*D + d;
                        int idx_k = b*N*H*D + j*H*D + h*D + d;
                        dot += Q.data_[idx_q] * K.data_[idx_k];
                    }
                    dot *= scale;
                    scores.push_back(dot);
                    if (dot > max_score) max_score = dot;
                }
                // 2. Softmax
                float sum_exp = 0.0f;
                for (size_t j = 0; j < scores.size(); ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                
                // 3. 加权求和 (Scores * V)
                for (int d = 0; d < D; ++d) {
                    float val = 0.0f;
                    for (int j = 0; j <= i; ++j) {
                        int idx_v = b*N*H*D + j*H*D + h*D + d;
                        val += (scores[j] / sum_exp) * V.data_[idx_v];
                    }
                    int idx_out = b*N*H*D + i*H*D + h*D + d;
                    out.data_[idx_out] = val;
                }
            }
        }
    }
    return out;
}

void test_attention() {
    std::cout << "=== Testing MultiHeadAttention ===" << std::endl;
    // --- 1. 正确性校验 (Small Scale) ---
    // Batch=1, Seq=16, Heads=2, Dim=32
    int B=1, N=16, H=2, D=32;
    Tensor Q({B, N, H, D});
    Tensor K({B, N, H, D}); 
    Tensor V({B, N, H, D});
    randomize_tensor(Q);
    randomize_tensor(K);
    randomize_tensor(V);
    // 运行你的优化版本
    Tensor out_opt = MultiHeadAttention::Prefill(Q, K, V);
    // 运行朴素版本
    Tensor out_ref = naive_attention(Q, K, V);
    // 比较 (容差稍微大一点，因为 Online Softmax 和标准 Softmax 浮点累加顺序不同)
    if (check_close(out_opt, out_ref, 1e-3, 1e-3)) {
        std::cout << "[PASS] Attention Logic (Online Softmax vs Naive)" << std::endl;
    } else {
        std::cout << "[FAIL] Attention logic mismatch!" << std::endl;
        // 打印前几个数来看看
        std::cout << "Opt[0]: " << out_opt.data_[0] << " Ref[0]: " << out_ref.data_[0] << std::endl;
        return; // 错误直接退出，不测速了
    }
    // --- 2. 性能测试 (Benchmark) ---
    // 模拟 Llama-2-7B 的部分参数规模
    // Dim = 128 (4096 / 32 heads)
    // Heads = 32
    // Batch = 1 (推理场景)
    // Seq Len: 测试短序列(128) 和 中长序列(1024)
    
    int bench_H = 24;
    int bench_D = 64;
    int bench_B = 1;
    
    std::vector<int> seq_lens = {32, 64, 128, 256, 384, 512, 768};
    for (int seq : seq_lens) {
        Tensor bQ({bench_B, seq, bench_H, bench_D});
        Tensor bK({bench_B, seq, bench_H, bench_D});
        Tensor bV({bench_B, seq, bench_H, bench_D});
        randomize_tensor(bQ);
        randomize_tensor(bK);
        randomize_tensor(bV);
        // 计算量估算 (FLOPs)
        // 1. Q * K^T: 2 * B * H * N^2 * D
        // 2. Attn * V: 2 * B * H * N^2 * D
        // (Softmax 和 Scale 忽略不计)
        // Total = 4 * B * H * N^2 * D
        double total_flops = 4.0 * bench_B * bench_H * std::pow(seq, 2) * bench_D;
        std::string name = "Attn [1," + std::to_string(seq) + ",24,64]";
        
        // 迭代次数随 seq 增加而减少，避免跑太久
        int iters = (seq > 512) ? 10 : 50; 
        benchmark(name, total_flops, iters, [&](){
            MultiHeadAttention::Prefill(bQ, bK, bV);
        });
    }
}

 **/
// --- 辅助：朴素 Attention 实现 (用于生成标准答案) ---
// 输入: [B, Seq, Heads, Dim]
Tensor naive_attention_ref(const Tensor& Q, const Tensor& K, const Tensor& V) {
    int B = Q.shape_[0];
    int N = Q.shape_[1];
    int H = Q.shape_[2];
    int D = Q.shape_[3];
    Tensor out(Q.shape_);
    float scale = 1.0f / std::sqrt((float)D);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < N; ++i) { 
                // 1. Scores
                std::vector<float> scores;
                float max_score = -1e9;
                for (int j = 0; j <= i; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        int idx_q = b*N*H*D + i*H*D + h*D + d;
                        int idx_k = b*N*H*D + j*H*D + h*D + d;
                        dot += Q.data_[idx_q] * K.data_[idx_k];
                    }
                    dot *= scale;
                    scores.push_back(dot);
                    if (dot > max_score) max_score = dot;
                }
                // 2. Softmax
                float sum_exp = 0.0f;
                for (size_t j = 0; j < scores.size(); ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                // 3. Output
                for (int d = 0; d < D; ++d) {
                    float val = 0.0f;
                    for (int j = 0; j <= i; ++j) {
                        int idx_v = b*N*H*D + j*H*D + h*D + d;
                        val += (scores[j] / sum_exp) * V.data_[idx_v];
                    }
                    int idx_out = b*N*H*D + i*H*D + h*D + d;
                    out.data_[idx_out] = val;
                }
            }
        }
    }
    return out;
}

void test_attention_module() {
    std::cout << "=== Testing Attention Module (0.7B Config) ===" << std::endl;
    
    // 0.7B 模型参数
    int HEADS = 24;
    int LATENT_DIM = 1536;
    int HEAD_DIM = LATENT_DIM / HEADS; // 64
    int MAX_SEQ = 768;
    int B = 1; // 推理时 Batch 通常为 1

    // ==========================================
    // 1. 正确性验证 (Prefill + Decode 联合验证)
    // ==========================================
    {
        std::cout << "[Checking Correctness]..." << std::endl;
        // 场景：先 Prefill 10 个 token，然后 Decode 第 11 个 token
        int prefill_len = 10;
        
        // 构造完整序列 (11个token) 的 Q, K, V 用于生成“标准答案”
        Tensor full_Q({B, prefill_len + 1, HEADS, HEAD_DIM});
        Tensor full_K({B, prefill_len + 1, HEADS, HEAD_DIM});
        Tensor full_V({B, prefill_len + 1, HEADS, HEAD_DIM});
        randomize_tensor(full_Q);
        randomize_tensor(full_K);
        randomize_tensor(full_V);

        // --- A. 计算标准答案 ---
        Tensor expected_full = naive_attention_ref(full_Q, full_K, full_V);
        
        // --- B. 测试 Prefill ---
        KVcache kv(MAX_SEQ, HEAD_DIM, HEADS); // 注意你的构造函数顺序
        
        // 切分出前 10 个 token
        // 注意：这里需要手动拷贝数据到新 Tensor，因为 Tensor 类不支持 view
        Tensor pre_Q({B, prefill_len, HEADS, HEAD_DIM});
        Tensor pre_K({B, prefill_len, HEADS, HEAD_DIM});
        Tensor pre_V({B, prefill_len, HEADS, HEAD_DIM});
        // 简单拷贝数据 (假设内存连续，直接用 size 比例拷贝)
        int step_size = HEADS * HEAD_DIM;
        for(int i=0; i<prefill_len * step_size; ++i) {
            pre_Q.data_[i] = full_Q.data_[i];
            pre_K.data_[i] = full_K.data_[i];
            pre_V.data_[i] = full_V.data_[i];
        }

        Tensor pre_out = MultiHeadAttention::prefill(pre_Q, pre_K, pre_V, kv);

        // 验证 Prefill 结果 (对比 expected_full 的前 10 行)
        // 抽样对比第 9 个 token 的第 0 个 head
        float val_opt = pre_out.data_[9 * step_size]; 
        float val_ref = expected_full.data_[9 * step_size];
        if (std::abs(val_opt - val_ref) < 1e-3) {
            std::cout << "  Prefill Logic: PASS" << std::endl;
        } else {
            std::cout << "  Prefill Logic: FAIL (Got " << val_opt << ", Expected " << val_ref << ")" << std::endl;
        }

        // --- C. 测试 Decode (Forward) ---
        // 准备第 11 个 token (index 10)
        Tensor next_Q({HEADS, HEAD_DIM});
        Tensor next_K({HEADS, HEAD_DIM});
        Tensor next_V({HEADS, HEAD_DIM});
        
        int offset = prefill_len * step_size;
        for(int i=0; i<step_size; ++i) {
            next_Q.data_[i] = full_Q.data_[offset + i];
            next_K.data_[i] = full_K.data_[offset + i];
            next_V.data_[i] = full_V.data_[offset + i];
        }

        // 执行 Forward (kv.n 应该是 10)
        assert(kv.n == prefill_len);
        Tensor dec_out = MultiHeadAttention::forward(next_Q, next_K, next_V, kv);

        // 验证 Decode 结果 (对比 expected_full 的第 10 行)
        // 注意 dec_out 形状是 [B=1, 1, m, dim] 或者 [m, dim]，取决于你的实现细节
        // 你的 forward 返回 res(Q.shape)，Q是(m, dim)，所以返回也是(m, dim)
        
        float val_dec = dec_out.data_[0]; // 第 0 个 head, 第 0 维
        float val_ref_dec = expected_full.data_[offset]; // 对应全量计算的第 10 行
        
        if (std::abs(val_dec - val_ref_dec) < 1e-3) {
            std::cout << "  Decode  Logic: PASS" << std::endl;
        } else {
            std::cout << "  Decode  Logic: FAIL (Got " << val_dec << ", Expected " << val_ref_dec << ")" << std::endl;
        }
    }

    // ==========================================
    // 2. 性能测试 (Benchmark)
    // ==========================================
    std::cout << "\n[Benchmarking Performance]..." << std::endl;

    // --- Case 1: Prefill Speed (128 tokens) ---
    {
        int seq_len = 128;
        Tensor Q({B, seq_len, HEADS, HEAD_DIM});
        Tensor K({B, seq_len, HEADS, HEAD_DIM});
        Tensor V({B, seq_len, HEADS, HEAD_DIM});
        randomize_tensor(Q); randomize_tensor(K); randomize_tensor(V);
        
        // 每次测试新建 KV Cache 避免状态累积
        // FLOPs: 4 * B * H * N^2 * D
        double flops = 4.0 * B * HEADS * std::pow(seq_len, 2) * HEAD_DIM;
        
        benchmark("Prefill (Seq=128)", flops, 50, [&](){
            KVcache kv(MAX_SEQ, HEAD_DIM, HEADS); // 构造耗时忽略不计
            MultiHeadAttention::prefill(Q, K, V, kv);
        });
    }

    // --- Case 2: Prefill Speed (512 tokens) ---
    {
        int seq_len = 512;
        Tensor Q({B, seq_len, HEADS, HEAD_DIM});
        Tensor K({B, seq_len, HEADS, HEAD_DIM});
        Tensor V({B, seq_len, HEADS, HEAD_DIM});
        randomize_tensor(Q); randomize_tensor(K); randomize_tensor(V);
        
        double flops = 4.0 * B * HEADS * std::pow(seq_len, 2) * HEAD_DIM;
        
        benchmark("Prefill (Seq=512)", flops, 10, [&](){
            KVcache kv(MAX_SEQ, HEAD_DIM, HEADS);
            MultiHeadAttention::prefill(Q, K, V, kv);
        });
    }

    // --- Case 3: Decode Speed (Cache Hit) ---
    // 模拟已经生成了 512 个 token，现在生成第 513 个
    {
        int context_len = 512;
        KVcache kv(MAX_SEQ, HEAD_DIM, HEADS);
        
        // 1. 先用 Prefill 填充 KV Cache 到 512 长度
        Tensor Q_pre({B, context_len, HEADS, HEAD_DIM});
        Tensor K_pre({B, context_len, HEADS, HEAD_DIM});
        Tensor V_pre({B, context_len, HEADS, HEAD_DIM});
        randomize_tensor(Q_pre); randomize_tensor(K_pre); randomize_tensor(V_pre);
        MultiHeadAttention::prefill(Q_pre, K_pre, V_pre, kv);

        // 2. 准备 Decode 用的输入 (单步)
        Tensor Q_step({HEADS, HEAD_DIM});
        Tensor K_step({HEADS, HEAD_DIM});
        Tensor V_step({HEADS, HEAD_DIM});
        randomize_tensor(Q_step); randomize_tensor(K_step); randomize_tensor(V_step);

        // Decode FLOPs 计算:
        // Q * K_cache^T: H * (Seq * D) 乘加 -> 2 * H * Seq * D
        // Attn * V_cache: H * (Seq * D) 乘加 -> 2 * H * Seq * D
        // Total = 4 * H * Seq * D
        // 这里 Seq = context_len (因为我们要和之前的 512 个历史做 Attention)
        double flops = 4.0 * HEADS * context_len * HEAD_DIM;

        // 注意：这里 benchmark 会循环多次，导致 kv.n 不断增加
        // 为了测速准确，我们不重置 kv，让它模拟 512 -> 612 的过程
        // 误差在允许范围内
        benchmark("Decode (Ctx=512)", flops, 100, [&](){
            MultiHeadAttention::forward(Q_step, K_step, V_step, kv);
        });
    }
}


int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "   Inference Engine Ops Benchmark (C++)" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;

    std::cout << "Warming up CPU..." << std::endl;
    Tensor dummy({2048, 2048});
    for(int i=0; i<10; i++) {
        Functions::matmul(dummy, dummy); // 把频率拉满
    }
    
    test_elementwise();
    test_softmax();
    test_rmsnorm();
    test_rope();
    test_matmul();
//    test_attention();
    test_attention_module();

    return 0;
}
