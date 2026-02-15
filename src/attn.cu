#include "tensor.h"
#include "utils.h"
#include "ops.h"
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cassert>
#include <immintrin.h>

KVcache::KVcache(int max_len, int dim, int heads, const std::string& device) :
    K_({max_len, heads, dim}, device),
    V_({max_len, heads, dim}, device),
    n(0),
    device_(device) {
}
KVcache::~KVcache() {}

void KVcache::to(const std::string& device) {
    if (device_ == device) return;
    if (device == "cpu") {
        K_.to("cpu");
        V_.to("cpu");
    }
    else if (device == "cuda") {
        K_.to("cuda");
        V_.to("cuda");
    }
    else {
        std::cerr << "KVcache : Invalid device: " << device << std::endl;
        exit(1);
    }
    device_ = device;
}

std::string KVcache::device() const {
    return device_;
}

const int BS = 16; // BLOCK_SIZE


// prefill 阶段，所有输入输出都是 (N, m, C) (实际的 C = 64)
// 考虑分块策略，所有结果在m维度上是并行的，所以令 bx 对应 m 
// by * BS + ty 处理 N , tx 对应 C
// 意思是，每个位置的最终结果都在一个block内计算完成，tx要算多次（C / BS），且一次处理 BS 个位置
__global__ void multihead_attn_prefill_kernel(
    const float* Q, 
    const float* K, 
    const float* V, 
    float* output,
    float* KV_K,
    float* KV_V,
    int N, int m, int C
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    int by = blockIdx.y, ty = threadIdx.y;
    float s_res[4] = {0.0f}; // 累加 res 时，tx在C维度并行，互不冲突，所以只需记录 C/BS 个元素即可
    __shared__ float s_max_val[BS][BS + 1];
    __shared__ float s_sum_exp[BS][BS + 1];
    __shared__ float s_dot_val[BS][BS + 1]; // dot[tx][ty]储存row[i + tx]和row[by * BS + ty]的dot
    __shared__ float s_q[BS][BS + 1];
    __shared__ float s_k[BS][BS + 1];
    __shared__ float s_v[BS][BS + 1];
    s_max_val[ty][tx] = -1e6f;
    s_sum_exp[ty][tx] = 0.0f;
    __syncthreads();
    float max_val = -1e6f, sum_exp = 0.0f;
    float scale = 1.0f / sqrtf(C);
    for (int i=0; i<=by; i++) {
        // step1: 分块计算块间的交叉dot
    //    s_dot_val[tx][ty] = 0.0f;
        float dot_val = 0.0f;
        bool avail_txty = ((i < by || tx <= ty) && (by * BS + ty < N) && (i * BS + tx < N));
        for (int j=0; j<C; j+=BS) {
            if (by * BS + ty < N) {
                s_q[ty][tx] = Q[(by * BS + ty) * m * C + bx * C + j + tx];
            }
            if (i * BS + ty < N) {
                s_k[ty][tx] = K[(i  * BS + ty) * m * C + bx * C + j + tx];
            }
            __syncthreads();
            if (avail_txty){
                for (int k=0; k<BS; k++) {
                    dot_val += s_q[ty][k] * s_k[tx][k] * scale;
                }
            }
            __syncthreads();
        }
        // step2: 根据块间交叉dot更新online softmax中的sum和max_val
        if (avail_txty) {
            float max_val = s_max_val[tx][ty], sum_exp = s_sum_exp[tx][ty];
            float new_max_val = fmaxf(max_val, dot_val);
            s_max_val[tx][ty] = new_max_val;
            s_sum_exp[tx][ty] = sum_exp * expf(max_val - new_max_val) + expf(dot_val - new_max_val);
        }
        __syncthreads();
        s_dot_val[tx][ty] = dot_val;
        // step3: 规约出当前经历的块的max_val，并维持sum_exp的语义不变
        // "保持语义不变"的意思是，sum_exp的放缩比例随对应的max_val改变，但16个sum_exp还是独立计算各自的加和
        for (int s=BS/2; s>0; s>>=1) {
            if (tx < s) {
                if (avail_txty) {
                    float tmp0 = s_max_val[tx + s * 0][ty];
                    float tmp1 = s_max_val[tx + s * 1][ty];
                    if (tmp1 > tmp0) {
                        s_max_val[tx][ty] = tmp1;
                        s_sum_exp[tx][ty] *= expf(tmp0 - tmp1);
                    }
                }
            }
            __syncthreads();
        }
        // step4: 利用max_val更新res (online softmax)
        // 此处，i 是固定的（对应第i块），j 枚举 C 维度处理到第几块了
        float new_max_val = s_max_val[0][ty];
        for (int j=0; j<C; j+=BS) {
            
            if (i * BS + ty < N) {
                s_v[ty][tx] = V[(i * BS + ty) * m * C + bx * C + j + tx];
            //    r_v[ty] = V[(i * BS + ty) * m * C + bx * C + j + tx];
            }
            __syncthreads();
            // k 处理第 i 块的第 k 个索引，ty 处理第 by 块的第 ty 个索引
            // 上述 k 的作用不能被 tx 取代，因为我们要做累加
            // 所以，tx 处理的是 C 维度第 j 块的第 tx 个元素
            float co_res = expf(max_val - new_max_val);
            s_res[j / BS] *= co_res;
            for (int k=0; k<BS; k++) {
                bool avail_kty = ((i < by || k <= ty) && (by * BS + ty < N) && (i * BS + k < N));
                if (avail_kty) {
                    float co_v = expf(s_dot_val[k][ty] - new_max_val);
                    s_res[j / BS] += s_v[k][tx] * co_v;
                //    s_res[j / BS] += r_v[k] * co_v;
                }
            }
            __syncthreads();
        }
        max_val = new_max_val;
    }
    // step5: 规约最终的sum_exp
    for (int s=BS/2; s>0; s>>=1) {
        if (tx < s) {
            float coef0 = expf(s_max_val[tx + s * 0][ty] - max_val);
            float coef1 = expf(s_max_val[tx + s * 1][ty] - max_val);
            s_sum_exp[tx][ty] = s_sum_exp[tx][ty] * coef0 + s_sum_exp[tx + s][ty] * coef1;
            s_max_val[tx][ty] = max_val;
        }
        __syncthreads();
    }
    sum_exp = s_sum_exp[0][ty];
    // step6: 写回结果
    for (int i=0; i<C; i+=BS) { 
        if (by * BS + ty < N) {
            float res = s_res[i / BS] / sum_exp;
        //    if (by == 1) res = sum_exp;
            output[(by * BS + ty) * m * C + bx * C + i + tx] = res;
        }
    }
    // step7: 填充KVcache
    if (by * BS + ty < N) {
        for (int i=0; i<C; i+=BS) {
            KV_K[(by * BS + ty) * m * C + bx * C + i + tx] = K[(by * BS + ty) * m * C + bx * C + i + tx];
            KV_V[(by * BS + ty) * m * C + bx * C + i + tx] = V[(by * BS + ty) * m * C + bx * C + i + tx];
        }
    }
}

// avx2加速的向量点乘
float __scale_dot(const float* idx_a, const float* idx_b, int n, float inv_sqrt_n) {
    assert(n % 8 == 0); // 只接受8的倍数的情况
    __m256 sum_vec = _mm256_setzero_ps();
    for (int j=0; j<n; j+=8) {
        __m256 a_vec = _mm256_loadu_ps(idx_a + j);
        __m256 b_vec = _mm256_loadu_ps(idx_b + j);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }
    float sum_val[8];
    _mm256_storeu_ps(sum_val, sum_vec);
    float sum = 0.0;
    for (int k=0; k<8; k++) {
        sum += sum_val[k];
    }
    return sum * inv_sqrt_n;
}

// a <- a * m + b * k
void __scale_add(float* idx_a, const float* idx_b, float m, float k, int n) {
    assert(n % 8 == 0); // 只接受8的倍数的情况
    __m256 vec_k = _mm256_set1_ps(k);
    __m256 vec_m = _mm256_set1_ps(m);
    for (int j=0; j<n; j+=8) {
        __m256 a_vec = _mm256_loadu_ps(idx_a + j);
        __m256 b_vec = _mm256_loadu_ps(idx_b + j);
        a_vec = _mm256_mul_ps(a_vec, vec_m);
        a_vec = _mm256_fmadd_ps(vec_k, b_vec, a_vec);
        _mm256_storeu_ps(idx_a + j, a_vec);
    }
}

void prefill_cpu(
    const float* Q, 
    const float* K, 
    const float* V, 
    float* output,
    float* KV_K,
    float* KV_V,
    int N, int m, int C    
) {
    float inv_sqrt_C = 1.0f / std::sqrt(C);
    memset(output, 0, N * m * C * sizeof(float));
    # pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<m; i++) {
        for (int j=0; j<N; j++) {
            float max_val = -1e6f, sum_exp = 0.0f;
            int idx_Q = j * m * C + i * C;
            for (int k=0; k<=j; k++) { 
                int idx_K = k * m * C + i * C;
                float dot_val = __scale_dot(Q + idx_Q, K + idx_K, C, inv_sqrt_C);
                if (dot_val > max_val) {
                    float scale = std::exp(max_val - dot_val);
                    __scale_add(output + idx_Q, V + idx_K, scale, 1.0f, C);
                    max_val = dot_val;
                    sum_exp *= scale;
                    sum_exp += 1.0f;
                }
                else {
                    float val_exp = std::exp(dot_val - max_val);
                    sum_exp += val_exp;
                    __scale_add(output + idx_Q, V + idx_K, 1.0f, val_exp, C);
                }
            }
        //    std::cout<<"i = "<<i<<" j = "<<j<<" max_val = "<<max_val<<" sum_exp = "<<sum_exp<<std::endl;
            __scale_add(output + idx_Q, V + idx_Q, 1.0f / sum_exp, 0.0f, C);
            std::copy(K + idx_Q, K + idx_Q + C, KV_K + idx_Q);
            std::copy(V + idx_Q, V + idx_Q + C, KV_V + idx_Q);
        }
    }
}

// 朴素prefill的实现：
const int TILE_SIZE = 16;

__global__ void matmul(
    const float* input, 
    const float* weight, 
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
            int T = fminf(M - (bx + ty * 4), 4);
            for (int j=0; j<T; j++) {
                w_s[ty * 4 + j][tx] = weight[(bx + ty * 4 + j) * K + i * TILE_SIZE + tx];
            }
        }
        if (by + ty * 4 < N) {
            int T = fminf(N - (by + ty * 4), 4);
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
    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            if (bx + tx * 4 + i < M && by + ty * 4 + j < N) {
                output[(by + ty * 4 + j) * M + (bx + tx * 4 + i)] = sum[j][i];
            }
        }
    }
}

void prefill_cuda_flash(
    const float* Q, 
    const float* K, 
    const float* V, 
    float* output,
    float* KV_K,
    float* KV_V,
    int N, int m, int C    
) {
    dim3 grid(m, (N + BS - 1) / BS);
    dim3 block(BS, BS);
    multihead_attn_prefill_kernel<<<grid, block>>>(
        Q, K, V, output, KV_K, KV_V, N, m, C
    );
    CHECK_CUDA(cudaGetLastError());
}




void MultiHeadAttention::prefill(
    const Tensor& Q, 
    const Tensor& K, 
    const Tensor& V, 
    Tensor& output,
    KVcache& KV
) {
    if (Q.dims() !=3 || Q.shape_ != K.shape_ || Q.shape_ != V.shape_ || Q.shape_ != output.shape_) {
        std::cerr << "MultiHeadAttention::prefill : Input shapes do not match." << std::endl;
        __print_vec__(Q.shape_, "Q");
        __print_vec__(K.shape_, "K");
        __print_vec__(V.shape_, "V");
        __print_vec__(output.shape_, "output");
        exit(1);
    }
    int N = Q.shape_[0], m = Q.shape_[1], C = Q.shape_[2];
    if (KV.K_.shape_!=KV.V_.shape_ || KV.K_.dims()!=3 || KV.K_.shape_[0]<N || KV.K_.shape_[1]!=m || KV.K_.shape_[2]!=C) {
        std::cerr << "MultiHeadAttention::prefill : KV cache shape does not match." << std::endl;
        __print_vec__(Q.shape_, "Q");
        __print_vec__(K.shape_, "K");
        __print_vec__(V.shape_, "V");
        __print_vec__(output.shape_, "output");
        exit(1);
    }
    if (Q.device() == "cpu" && K.device() == "cpu" && V.device() == "cpu" && output.device() == "cpu" && KV.device() == "cpu") {
        prefill_cpu(
            Q.h_ptr, K.h_ptr, V.h_ptr,
            output.h_ptr, 
            KV.K_.h_ptr, KV.V_.h_ptr,
            N, m, C
        );
        KV.n = N;
    }
    else if (Q.device() == "cuda" && K.device() == "cuda" && V.device() == "cuda" && output.device() == "cuda" && KV.device() == "cuda") {
    //    std::cout<<"grid: "<<grid.x<<" "<<grid.y<<" block: "<<block.x<<" "<<block.y<<std::endl;
        prefill_cuda_flash(
            Q.d_ptr, K.d_ptr, V.d_ptr,
            output.d_ptr, 
            KV.K_.d_ptr, KV.V_.d_ptr,
            N, m, C
        );
        KV.n = N;
    }
    else {
        std::cerr << "MultiHeadAttention::prefill : Input devices do not match." << std::endl;
    }
}

const int BC = 64; // 一个block在 C维度的大小
const int BN = 16; // 一个block在 N维度的大小

// 这样一来，一个块的大小是 1024 > max_len , 规约时可以一次性处理掉
// bx 对应 m , tx 对应 C , ty 对应 N 的分块
__global__ void multihead_attn_forward_kernel(
    const float* Q, 
    const float* K, 
    const float* V, 
    float* output,
    float* KV_K,
    float* KV_V,
    int N, int m, int C
) {
    int bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
    int idx = ty * BC + tx;
    
    if (ty == 0) {
        KV_K[N * m * C + bx * C + tx] = K[bx * C + tx];
        KV_V[N * m * C + bx * C + tx] = V[bx * C + tx];
    }
    
    __shared__ float s_max_val[BN * BC];
    __shared__ float s_sum_exp[BN * BC];
    __shared__ float s_dot_val[BN][BC + 1];
    __shared__ float s_q[BC];
    s_max_val[idx] = -1000000.0f;
    s_sum_exp[idx] = 0.0f;
    if (ty == 0) {
        s_q[tx] = Q[bx * C + tx];
    }
    __syncthreads();
    float scale = 1.0f / sqrtf(C);
    for (int i=0; i <= N; i += BN) {
        if ((i + ty) <= N) {
            s_dot_val[ty][tx] = KV_K[(i + ty) * m * C + bx * C + tx] * s_q[tx];
        }
        __syncthreads();
        for (int s=BC/2; s>0; s>>=1) {
            if (tx < s) {
                s_dot_val[ty][tx] += s_dot_val[ty][tx + s];
            }
            __syncthreads();
        }
        if (tx == 0 && i + ty <= N) {
            float dot_val = s_dot_val[ty][0] * scale;
            s_max_val[i + ty] = dot_val;
            s_sum_exp[i + ty] = 1.0f;
        }
    }
    __syncthreads();
    s_dot_val[ty][tx] = s_max_val[idx]; // 这里 s_dot_val 语义和之前不同，现在是暂存 每个位置的dot
    for (int s=BN*BC/2; s>0; s>>=1) {
        if (idx < s) {
            float tmp0 = s_max_val[idx], tmp1 = s_max_val[idx + s];
            float new_max_val = fmaxf(tmp0, tmp1);
            s_sum_exp[idx] = s_sum_exp[idx    ] * expf(tmp0 - new_max_val) +
                             s_sum_exp[idx + s] * expf(tmp1 - new_max_val);
            s_max_val[idx] = new_max_val;
        }
        __syncthreads();
    }
    float max_val = s_max_val[0];
    float sum_exp = s_sum_exp[0];
    __syncthreads();
    s_sum_exp[idx] = 0.0f; // 之后用不到sum_exp了，直接复用其显存来存res
    for (int i=0; i<=N; i+=BN) {
        if ((i + ty) <= N) {
            float coef = expf(s_dot_val[(i + ty) / BC][(i + ty) % BC] - max_val) / sum_exp;
            s_sum_exp[idx] += coef * KV_V[(i + ty) * m * C + bx * C + tx];
        }
    }
    __syncthreads();
    for (int s=BN/2; s>0; s>>=1) {
        if (ty < s) {
            s_sum_exp[ty * BC + tx] += s_sum_exp[(ty + s) * BC + tx];
        }
        __syncthreads();
    }
    output[bx * C + tx] = s_sum_exp[0 * BC + tx];
}

void forward_cpu(
    const float* Q, 
    const float* K, 
    const float* V, 
    float* output,
    float* KV_K,
    float* KV_V,
    int N, int m, int C
) {
    float inv_sqrt_C = 1.0f / sqrtf(C);
    memset(output, 0, m * C * sizeof(float));
    # pragma omp parallel for schedule(static)
    for (int i=0; i<m; i++) {
        int idx_KV = N * m * C + i * C;
        int idx_Q = i * C;
        std::copy(K + idx_Q, K + idx_Q + C, KV_K + idx_KV);
        std::copy(V + idx_Q, V + idx_Q + C, KV_V + idx_KV);
        float max_val = -1000000.0, sum_exp = 0.0;
        for (int j=0; j<=N; j++) {
            int idx_K = j * m * C + i * C;
            float dot_val = __scale_dot(Q + idx_Q, KV_K + idx_K, C, inv_sqrt_C);
            if (dot_val > max_val) {
                float scale = expf(max_val - dot_val);
                __scale_add(output + idx_Q, KV_V + idx_K, scale, 1.0f, C);
                sum_exp *= scale;
                sum_exp += 1.0f;
                max_val = dot_val;
            }
            else {
                float exp_val = expf(dot_val - max_val);
                sum_exp += exp_val;
                __scale_add(output + idx_Q, KV_V + idx_K, 1.0f, exp_val, C);
            }
        }
        __scale_add(output + idx_Q, KV_V + idx_KV, 1.0f / sum_exp, 0.0f, C);
    }

}



void MultiHeadAttention::forward(
    const Tensor& Q, 
    const Tensor& K, 
    const Tensor& V, 
    Tensor& output,
    KVcache& KV
) {
    if (Q.dims() !=2 || Q.shape_ != K.shape_ || Q.shape_ != V.shape_ || Q.shape_ != output.shape_) {
        std::cerr << "MultiHeadAttention::forward : Input shapes do not match." << std::endl;
        __print_vec__(Q.shape_, "Q");
        __print_vec__(K.shape_, "K");
        __print_vec__(V.shape_, "V");
        __print_vec__(output.shape_, "output");
        exit(1);
    }
    int m = Q.shape_[0], C = Q.shape_[1];
    if (KV.K_.shape_!=KV.V_.shape_ || KV.K_.dims()!=3 || KV.K_.shape_[0]<=KV.n || KV.K_.shape_[1]!=m || KV.K_.shape_[2]!=C) {
        std::cerr << "MultiHeadAttention::forward : KV cache shape does not match." << std::endl;
        __print_vec__(Q.shape_, "Q");
        __print_vec__(K.shape_, "K");
        __print_vec__(V.shape_, "V");
        __print_vec__(output.shape_, "output");
        __print_vec__(KV.K_.shape_, "KV.K_");
        __print_vec__(KV.V_.shape_, "KV.V_");
        std::cout<<"pos = "<<KV.n<<std::endl;
        exit(1);
    }
    if (Q.device() == "cpu" && K.device() == "cpu" && V.device() == "cpu" && output.device() == "cpu" && KV.device() == "cpu") {
        forward_cpu(
            Q.h_ptr, K.h_ptr, V.h_ptr,
            output.h_ptr,
            KV.K_.h_ptr, KV.V_.h_ptr,
            KV.n, m, C
        );
        KV.n++;
    }
    else if (Q.device() == "cuda" && K.device() == "cuda" && V.device() == "cuda" && output.device() == "cuda" && KV.device() == "cuda") {
        dim3 grid(m);
        dim3 block(BC, BN);
        multihead_attn_forward_kernel<<<grid, block>>>(
            Q.d_ptr, K.d_ptr, V.d_ptr,
            output.d_ptr,
            KV.K_.d_ptr, KV.V_.d_ptr,
            KV.n, m, C
        );
        CHECK_CUDA(cudaGetLastError());
        KV.n++;
    }
    else {
        std::cerr << "MultiHeadAttention::forward : Input devices do not match." << std::endl;
        std::cout<<"Q.device="<<Q.device()<<std::endl;
        std::cout<<"K.device="<<K.device()<<std::endl;
        std::cout<<"V.device="<<V.device()<<std::endl;
        std::cout<<"output.device="<<output.device()<<std::endl;
        std::cout<<"KV.device="<<KV.device()<<std::endl;
    }

}

AttentionBlock::AttentionBlock(int max_len, int latent_dim, int m, float base_rope, std::string device) : 
    Q_(latent_dim, latent_dim, device),
    K_(latent_dim, latent_dim, device),
    V_(latent_dim, latent_dim, device),
    W_(latent_dim, latent_dim, device),
    workplace_Q({max_len, latent_dim}, device),
    workplace_K({max_len, latent_dim}, device),
    workplace_V({max_len, latent_dim}, device),
    workplace_W({max_len, latent_dim}, device),
    vec_Q({latent_dim}, device),
    vec_K({latent_dim}, device),
    vec_V({latent_dim}, device),
    vec_W({latent_dim}, device),
    KV(max_len, latent_dim / m, m, device),
    max_len(max_len),
    latent_dim(latent_dim),
    m(m),
    device_(device), 
    base_rope(base_rope) {
}

AttentionBlock::~AttentionBlock() {}

void AttentionBlock::load(const std::string& base_path) {
    Q_.load(base_path + "/Q");
    K_.load(base_path + "/K");
    V_.load(base_path + "/V");
    W_.load(base_path + "/W");
}

std::string AttentionBlock::device() const {
    return device_;
}

void AttentionBlock::to(const std::string& device) {
    if (device_ == device) return;
    if (device == "cpu") {
        Q_.to("cpu");
        K_.to("cpu");
        V_.to("cpu");
        W_.to("cpu");
        KV.to("cpu");
        workplace_K.to("cpu");
        workplace_Q.to("cpu");
        workplace_V.to("cpu");
        workplace_W.to("cpu");
        vec_K.to("cpu");
        vec_Q.to("cpu");
        vec_V.to("cpu");
        vec_W.to("cpu");
    }
    else if (device == "cuda") {
        Q_.to("cuda");
        K_.to("cuda");
        V_.to("cuda");
        W_.to("cuda");
        KV.to("cuda");
        workplace_K.to("cuda");
        workplace_Q.to("cuda");
        workplace_V.to("cuda");
        workplace_W.to("cuda");
        vec_K.to("cuda");
        vec_Q.to("cuda");
        vec_V.to("cuda");
        vec_W.to("cuda");
    }
    else {
        std::cerr << "AttentionBlock : Invalid device: " << device << std::endl;
        exit(1);
    }
    device_ = device;
}

void AttentionBlock::forward(const Tensor& x, Tensor& output, bool is_prefill) {
    if (is_prefill) {
        int N = x.shape_[0];
        int K = latent_dim;
        if (N > workplace_K.shape_[0] || N >=1024) {
            std::cerr << "AttentionBlock : prefill text too long " << x.shape_[0] << std::endl;
            exit(1);
        }
        workplace_Q.shape_ = {N, K}; Q_.forward(x, workplace_Q); workplace_Q.reshape({N, m, K / m});
        workplace_K.shape_ = {N, K}; K_.forward(x, workplace_K); workplace_K.reshape({N, m, K / m});
        workplace_V.shape_ = {N, K}; V_.forward(x, workplace_V); workplace_V.reshape({N, m, K / m});
        
        rope_.forward(workplace_Q, base_rope);
        rope_.forward(workplace_K, base_rope);

        workplace_W.shape_ = {N, m, K / m};
        MultiHeadAttention::prefill(workplace_Q, workplace_K, workplace_V, workplace_W, KV);
        
        workplace_W.reshape({N, K});
        W_.forward(workplace_W, output);
        
        workplace_Q.shape_ = {max_len, K};
        workplace_K.shape_ = {max_len, K};
        workplace_V.shape_ = {max_len, K};
        workplace_W.shape_ = {max_len, K};
    }
    else {
        int K = latent_dim;
        vec_Q.shape_ = {K}; Q_.forward_vec(x, vec_Q); vec_Q.reshape({m, K / m});
        vec_K.shape_ = {K}; K_.forward_vec(x, vec_K); vec_K.reshape({m, K / m});
        vec_V.shape_ = {K}; V_.forward_vec(x, vec_V); vec_V.reshape({m, K / m});

        rope_.forward_vec(vec_Q, KV.n, base_rope);
        rope_.forward_vec(vec_K, KV.n, base_rope);

        vec_W.shape_ = {m, K / m};
        ma_.forward(vec_Q, vec_K, vec_V, vec_W, KV);

        vec_W.reshape({K});
        W_.forward_vec(vec_W, output);

        
    }
}

