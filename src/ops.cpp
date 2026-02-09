#include "ops.h"
#include "tensor.h"
#include <iostream>
#include <cmath>
#include <string>
#include <memory>
#include <cassert>
#include <immintrin.h>
#include <cstring>



Tensor Functions::Exp(const Tensor& x) {
    Tensor res(x.shape_);
    int size_ = x.size();
    # pragma omp parallel for schedule(static)
    for (int i = 0; i < size_; i++) {
        res.data_[i] = std::exp(x.data_[i]);
    }
    return res;
}
Tensor Functions::Log(const Tensor& x) {
    Tensor res(x.shape_);
    int size_ = x.size();
    # pragma omp parallel for schedule(static)
    for (int i = 0; i < size_; i++) {
        res.data_[i] = std::log(x.data_[i]);
    }
    return res;
}
Tensor Functions::Sin(const Tensor& x) {
    Tensor res(x.shape_);
    int size_ = x.size();
    # pragma omp parallel for schedule(static)
    for (int i = 0; i < size_; i++) {
        res.data_[i] = std::sin(x.data_[i]);
    }
    return res;
}
Tensor Functions::Cos(const Tensor& x) {
    Tensor res(x.shape_);
    int size_ = x.size();
    # pragma omp parallel for schedule(static)
    for (int i = 0; i < size_; i++) {
        res.data_[i] = std::cos(x.data_[i]);
    }
    return res;
}
Tensor Functions::Sqrt(const Tensor& x) {
    Tensor res(x.shape_);
    int size_ = x.size();
    # pragma omp parallel for schedule(static)
    for (int i = 0; i < size_; i++) {
        res.data_[i] = std::sqrt(x.data_[i]);
    }
    return res;
}
Tensor Functions::Pow(const Tensor& x, float p) {
    Tensor res(x.shape_);
    int size_ = x.size();
    # pragma omp parallel for schedule(static)
    for (int i = 0; i < size_; i++) {
        res.data_[i] = std::pow(x.data_[i], p);
    }
    return res;
}

Tensor Functions::sigmoid(const Tensor& x) {
    Tensor res(x.shape_);
    int size_ = x.size();
    # pragma omp parallel for schedule(static)
    for (int i = 0; i < size_; i++) {
        res.data_[i] = 1.0 / (1.0 + std::exp(-x.data_[i]));
    }
    return res;
}
Tensor Functions::tanh(const Tensor& x) {
    Tensor res(x.shape_);
    int size_ = x.size();
    # pragma omp parallel for schedule(static)
    for (int i = 0; i < size_; i++) {
        res.data_[i] = std::tanh(x.data_[i]);
    }
    return res;
}
Tensor Functions::relu(const Tensor& x) {
    Tensor res(x.shape_);
    int size_ = x.size();
    # pragma omp parallel for schedule(static)
    for (int i = 0; i < size_; i++) {
        res.data_[i] = std::max(0.0f, x.data_[i]);
    }
    return res;
}
Tensor Functions::silu(const Tensor& x) {
    Tensor res(x.shape_);
    int size_ = x.size();
    # pragma omp parallel for schedule(static)
    for (int i = 0; i < size_; i++) {
        res.data_[i] = x.data_[i] / (1.0 + std::exp(-x.data_[i]));
    }
    return res;
}

// 仅对最后一维算softmax
Tensor Functions::softmax(const Tensor& x) {
    Tensor res = x.contiguous();
    int n = res.shape_[res.shape_.size() - 1];
    int size_ = res.size();
    # pragma omp parallel for schedule(static)
    for (int i=0; i < size_; i+=n) {
        float max_val = -1000000.0;
        for (int j=0; j<n; j++) {
            if (res.data_[i+j] > max_val) {
                max_val = res.data_[i+j];
            }
        }
        float sum_exp = 0.0;
        for (int j=0; j<n; j++) {
            res.data_[i+j] = std::exp(res.data_[i+j] - max_val);
            sum_exp += res.data_[i+j];
        }
        for (int j=0; j<n; j++) {
            res.data_[i+j] /= sum_exp;
        }
    }
    return res;
}

// 这里只考虑左矩阵乘右向量，即(a,b) * (...,b) -> (...,a)
Tensor Functions::matmul(const Tensor& A, const Tensor& B) {
    assert(A.shape_[A.shape_.size() - 1] == B.shape_[B.shape_.size() - 1]);
    Tensor a_c = A.contiguous();
    Tensor b_c = B.contiguous();
    std::vector<int> new_shape = B.shape_;
    new_shape[B.shape_.size() - 1] = A.shape_[0];
    Tensor res(new_shape);
    int n = A.shape_[A.shape_.size() - 1];
    assert(n % 8 == 0); // 只接受8的倍数的情况
    int A_size = A.size() / n;
    int B_size = B.size() / n;
    int res_size = res.size();
    # pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<A_size; i++) {
        for (int j=0; j<B_size; j++) {
            __m256 res_vec = _mm256_setzero_ps();
            for (int k=0; k<n; k+=8) {
                __m256 a_vec = _mm256_loadu_ps(&a_c.data_[i * n + k]);
                __m256 b_vec = _mm256_loadu_ps(&b_c.data_[j * n + k]);
                res_vec = _mm256_fmadd_ps(a_vec, b_vec, res_vec);
            }
            float res_val[8];
            _mm256_storeu_ps(res_val, res_vec);
            float sum = 0.0;
            for (int k=0; k<8; k++) {
                sum += res_val[k];
            }
            res.data_[i + j * A_size] = sum;
        }
    }
    return res;
}

void Functions::matmul_inplace(const Tensor& A, const Tensor& B, Tensor& res) {
    assert(A.shape_[A.shape_.size() - 1] == B.shape_[B.shape_.size() - 1]);
    Tensor a_c = A.contiguous();
    Tensor b_c = B.contiguous();
    std::vector<int> new_shape = B.shape_;
    new_shape[B.shape_.size() - 1] = A.shape_[0];
    assert(res.shape_ == new_shape);
    int n = A.shape_[A.shape_.size() - 1];
    assert(n % 8 == 0); // 只接受8的倍数的情况
    int A_size = A.size() / n;
    int B_size = B.size() / n;
    int res_size = res.size();
    # pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<A_size; i++) {
        for (int j=0; j<B_size; j++) {
            __m256 res_vec = _mm256_setzero_ps();
            for (int k=0; k<n; k+=8) {
                __m256 a_vec = _mm256_loadu_ps(&a_c.data_[i * n + k]);
                __m256 b_vec = _mm256_loadu_ps(&b_c.data_[j * n + k]);
                res_vec = _mm256_fmadd_ps(a_vec, b_vec, res_vec);
            }
            float res_val[8];
            _mm256_storeu_ps(res_val, res_vec);
            float sum = 0.0;
            for (int k=0; k<8; k++) {
                sum += res_val[k];
            }
            res.data_[i + j * A_size] = sum;
        }
    }
}

Tensor Functions::dot(const Tensor& A, const Tensor& B) {
    assert(A.shape_ == B.shape_);
    Tensor a_c = A.contiguous();
    Tensor b_c = B.contiguous();
    std::vector<int> new_shape = {A.shape_};
    new_shape.pop_back();
    Tensor res(new_shape);
    int n = A.shape_[A.shape_.size() - 1];
    assert(n % 8 == 0); // 只接受8的倍数的情况
    int size_ = A.size() / n;
    # pragma omp parallel for schedule(static)
    for (int i=0; i<size_; i++) {
        __m256 res_vec = _mm256_setzero_ps();
        for (int k=0; k<n; k+=8) {
            __m256 a_vec = _mm256_loadu_ps(&a_c.data_[i * n + k]);
            __m256 b_vec = _mm256_loadu_ps(&b_c.data_[i * n + k]);
            res_vec = _mm256_fmadd_ps(a_vec, b_vec, res_vec);
        }
        float res_val[8];
        _mm256_storeu_ps(res_val, res_vec);
        float sum = 0.0;
        for (int k=0; k<8; k++) {
            sum += res_val[k];
        }
        res.data_[i] = sum;
    }
    return res;
}

Linear::Linear(int in_dim, int out_dim) : 
    coef_({out_dim, in_dim}), bias_({out_dim}) {
    // 初始化创建矩阵
}

Linear::~Linear() {
    // miao~
}

void Linear::load(const std::string& base_path) {
    coef_.load(base_path + "/weight.bin");
    bias_.load(base_path + "/bias.bin");
}

Tensor Linear::forward(const Tensor& x) const {
    std::vector<int>broadcast_shape(x.shape_.size(), 1);
    broadcast_shape[x.shape_.size() - 1] = coef_.shape_[0];
    return Functions::matmul(coef_, x) + bias_.reshape(broadcast_shape);
}

void Linear::forward_inplace(const Tensor& x, Tensor& res) const {
    Functions::matmul_inplace(coef_, x, res);
    int dim = bias_.shape_[0];
    int B = res.size() / dim;
    # pragma omp parallel for schedule(static)
    for (int i=0; i<B; i++) {
        for (int j=0; j<dim; j++) {
            res.data_[i * dim + j] += bias_.data_[j];
        }
    }
}


RMSNorm::RMSNorm(int dim) : gamma_({dim}) {
    // miao~
}
RMSNorm::~RMSNorm() {
    // miao~
}    

void RMSNorm::load(const std::string& base_path) {
    gamma_.load(base_path + "/weight.bin");
}
Tensor RMSNorm::forward(const Tensor& input) const {
    Tensor x = input.contiguous().clone();

    int n = x.shape_[x.shape_.size() - 1];
    int size_ = x.size() / n;
    assert(n % 8 == 0); // 只接受8的倍数的情况

    # pragma omp parallel for schedule(static)
    for (int i=0; i<size_; i++) {
        __m256 sum_vec = _mm256_setzero_ps();
        for (int j=0; j<n; j+=8) {
            __m256 x_vec = _mm256_loadu_ps(&x.data_[i * n + j]);
            sum_vec = _mm256_fmadd_ps(x_vec, x_vec, sum_vec);
        }
        float sum_val[8];
        _mm256_storeu_ps(sum_val, sum_vec);
        float sum = 0.0;
        for (int k=0; k<8; k++) {
            sum += sum_val[k];
        }
        float sqrt_mean = std::sqrt(sum / n + 1e-6);
        float coef = 1.0 / sqrt_mean;
        for (int j=0; j<n; j+=8) {
            __m256 x_vec = _mm256_loadu_ps(&x.data_[i * n + j]);
            __m256 gamma_vec = _mm256_loadu_ps(&gamma_.data_[j]);
            x_vec = _mm256_mul_ps(x_vec, _mm256_set1_ps(coef));
            x_vec = _mm256_mul_ps(x_vec, gamma_vec);
            _mm256_storeu_ps(&x.data_[i * n + j], x_vec);
        }
    }
    return x;
}

RoPE::RoPE(float base = 10000.0, int max_len = 1024, int dim = 64) : base_(base), max_len_(max_len), dim_(dim), P({max_len, dim/2, 2}) {
    for (int i=0; i<max_len; i++) {
        for (int j=0; j<dim/2; j++) {
            float theta = pow(base, - 2.0f * j / dim) * i;
            P.data_[i * dim + j * 2 + 0] = std::cos(theta);
            P.data_[i * dim + j * 2 + 1] = std::sin(theta);
        }
    }
}
RoPE::~RoPE() {
    // miao~
}

void RoPE::forward(Tensor& x) const {
    x = x.contiguous();
    assert(x.dims() >= 3);
    assert(x.shape_[x.shape_.size() - 1] == dim_);
    assert(x.shape_[x.shape_.size() - 3] <= max_len_);
    assert(dim_ % 2 == 0);
    int n = x.shape_[x.shape_.size() - 3];
    int m = x.shape_[x.shape_.size() - 2];
    int B = x.size() / (n * m * dim_);
    # pragma omp parallel for collapse(3) schedule(static)
    for (int i=0; i<B; i++) {
        for (int j=0; j<n; j++) {
            for (int k=0; k<m; k++) {
                for (int l=0; l<dim_/2; l++) {
                    int idx_x = i * n * m * dim_ + j * m * dim_ + k * dim_ + l;
                    int idx_P = j * dim_ + l * 2;
                    float cosP = P.data_[idx_P + 0];
                    float sinP = P.data_[idx_P + 1];
                    float x0_rope =  x.data_[idx_x] * cosP + x.data_[idx_x + dim_/2] * sinP;
                    float x1_rope = -x.data_[idx_x] * sinP + x.data_[idx_x + dim_/2] * cosP;
                    x.data_[idx_x] = x0_rope;
                    x.data_[idx_x + dim_/2] = x1_rope;
                }
            }
        }
    }
}

void RoPE::forward(Tensor& x, int pos) const {
    x = x.contiguous();
    assert(x.dims() == 2);
    assert(x.shape_[x.shape_.size() - 1] == dim_);
    assert(pos <= max_len_);
    assert(dim_ % 2 == 0);
    int m = x.shape_[x.shape_.size() - 2];

    # pragma omp parallel for schedule(static)
    for (int i=0; i<m; i++) {
        for (int j=0; j<dim_/2; j++) {
            int idx_x = i * dim_ + j;
            int idx_P = pos * dim_ + j * 2;
            float cosP = P.data_[idx_P + 0];
            float sinP = P.data_[idx_P + 1];
            float x0_rope =  x.data_[idx_x] * cosP + x.data_[idx_x + dim_/2] * sinP;
            float x1_rope = -x.data_[idx_x] * sinP + x.data_[idx_x + dim_/2] * cosP;
            x.data_[idx_x] = x0_rope;
            x.data_[idx_x + dim_/2] = x1_rope;
        }
    }
}

SwishGLU::SwishGLU(int in_dim, int hidden_dim) : W1_(in_dim, hidden_dim), W2_(hidden_dim, in_dim), V_(in_dim, hidden_dim) {
    // miao~
}
SwishGLU::~SwishGLU() {
    // miao~
}

void SwishGLU::load(const std::string& base_path) {
    W1_.load(base_path + "/W1");
    W2_.load(base_path + "/W2");
    V_.load(base_path + "/V");
}
Tensor SwishGLU::forward(const Tensor& x) const {
    return W2_.forward(Functions::silu(W1_.forward(x)) * V_.forward(x));
}

// avx2加速的向量点乘
float __scale_dot(float* idx_a, float* idx_b, int n, float inv_sqrt_n) {
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
void __scale_add(float* idx_a, float* idx_b, float m, float k, int n) {
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

KVcache::KVcache(int max_len, int dim, int heads) : K_({max_len, heads, dim}), V_({max_len, heads, dim}), n(0) {
    // miao~
}
KVcache::~KVcache() {
    // miao~
}



Tensor MultiHeadAttention::prefill(const Tensor& Q, const Tensor& K, const Tensor& V, KVcache& KV) {
    assert(Q.shape_ == K.shape_);
    assert(K.shape_ == V.shape_);
    assert(Q.dims() >= 3);
    Tensor Q_c = Q.contiguous();
    Tensor K_c = K.contiguous();
    Tensor V_c = V.contiguous();
    
    int dim = Q.shape_[Q.shape_.size() - 1];
    int m = Q.shape_[Q.shape_.size() - 2];
    int n = Q.shape_[Q.shape_.size() - 3];
    int B = Q.size() / (n * m * dim);
    float inv_sqrt_n = 1.0 / std::sqrt(float(dim));
    Tensor res(Q_c.shape_);
    memset(&res.data_[0], 0, res.size() * sizeof(float));

    // 初始化 KVcache
    int max_len = KV.K_.shape_[0];
    assert(KV.K_.shape_ == KV.V_.shape_);
    assert(KV.K_.dims() == 3);
    assert(max_len >= n);
    assert(KV.K_.shape_[1] == m);
    assert(KV.K_.shape_[2] == dim);
    KV.n = n;
    
    // 手动模拟softmax
    # pragma omp parallel for collapse(3) schedule(static)
    for (int i=0; i<B; i++) {
        for (int k=0; k<m; k++) {
            for (int j=0; j<n; j++) {
                float max_val = -1000000.0, sum_exp = 0.0;
                int idx_Q = i * n * m * dim + j * dim * m + k * dim;
                for (int l=0; l<=j; l++) {  // 启用因果掩码, l <= j
                    int idx_K = i * n * m * dim + l * dim * m + k * dim;
                    float dot_val = __scale_dot(&Q_c.data_[0] + idx_Q, &K_c.data_[0] + idx_K, dim, inv_sqrt_n);
                    if (dot_val > max_val) {
                        float scale_m = std::exp(max_val - dot_val);
                        sum_exp *= scale_m;
                        sum_exp ++ ;
                        max_val = dot_val;
                        __scale_add(&res.data_[0] + idx_Q, &V_c.data_[0] + idx_K, scale_m, 1.0, dim);
                    }
                    else {
                        float val_exp = std::exp(dot_val - max_val);
                        sum_exp += val_exp;
                        __scale_add(&res.data_[0] + idx_Q, &V_c.data_[0] + idx_K, 1.0, val_exp, dim);
                    }
                }
                __scale_add(&res.data_[0] + idx_Q, &res.data_[0] + idx_Q, 1.0 / sum_exp, 0.0, dim);

                // 保存 KV
                int idx_KV = j * m * dim + k * dim;
                std::copy(&K_c.data_[idx_Q], &K_c.data_[idx_Q] + dim, &KV.K_.data_[idx_KV]);
                std::copy(&V_c.data_[idx_Q], &V_c.data_[idx_Q] + dim, &KV.V_.data_[idx_KV]);
            }
        }
    }
    return res;
}

Tensor MultiHeadAttention::forward(const Tensor& Q, const Tensor& K, const Tensor& V, KVcache& KV) {
    assert(Q.shape_ == K.shape_);
    assert(K.shape_ == V.shape_);
    assert(Q.dims() == 2);
    Tensor Q_c = Q.contiguous();
    Tensor K_c = K.contiguous();
    Tensor V_c = V.contiguous();

    int dim = Q.shape_[Q.shape_.size() - 1];
    int m = Q.shape_[0];
    int n = KV.n;
    int max_len = KV.K_.shape_[0];
    float inv_sqrt_n = 1.0 / std::sqrt(float(dim));
 //   std::cout<<"max_len: "<<max_len<<" n:"<<n<<std::endl;
    assert(max_len > n);
    assert(KV.K_.shape_ == KV.V_.shape_);
    assert(KV.K_.dims() == 3);
    assert(KV.K_.shape_[1] == m);
    assert(KV.K_.shape_[2] == dim);
    assert(dim % 8 == 0); // 只接受8的倍数的情况

    Tensor res(Q_c.shape_);
    memset(&res.data_[0], 0, res.size() * sizeof(float));

    # pragma omp parallel for schedule(static)
    for (int i=0; i<m; i++) {
        int idx_KV = n * m * dim + i * dim;
        int idx_Q = i * dim;
        std::copy(&K_c.data_[idx_Q], &K_c.data_[idx_Q] + dim, &KV.K_.data_[idx_KV]);
        std::copy(&V_c.data_[idx_Q], &V_c.data_[idx_Q] + dim, &KV.V_.data_[idx_KV]);

        
        float max_val = -1000000.0, sum_exp = 0.0;
        for (int j=0; j<=n; j++) {
            int idx_K = j * dim * m + i * dim;
            float dot_val = __scale_dot(&Q_c.data_[0] + idx_Q, &KV.K_.data_[0] + idx_K, dim, inv_sqrt_n);
            if (dot_val > max_val) {
                float scale_m = std::exp(max_val - dot_val);
                sum_exp *= scale_m;
                sum_exp ++ ;
                max_val = dot_val;
                __scale_add(&res.data_[0] + idx_Q, &KV.V_.data_[0] + idx_K, scale_m, 1.0, dim);
            }
            else {
                float val_exp = std::exp(dot_val - max_val);
                sum_exp += val_exp;
                __scale_add(&res.data_[0] + idx_Q, &KV.V_.data_[0] + idx_K, 1.0, val_exp, dim);
            }
        }
        __scale_add(&res.data_[0] + idx_Q, &res.data_[0] + idx_Q, 1.0 / sum_exp, 0.0, dim);
    }
    KV.n ++ ;
    return res;
}

Embedding::Embedding(int vocabulary, int latent_dim) : 
    vocabulary(vocabulary), latent_dim(latent_dim), weight_({vocabulary, latent_dim}), workplace_({vocabulary}) {
    // miao~
}
Embedding::~Embedding() {
    // miao~
}

void Embedding::load(const std::string& base_path) {
    weight_.load(base_path + "/weight.bin");
}
Tensor Embedding::forward(const std::vector<int>& tokens) const {
    int n = tokens.size();
    Tensor res({n, latent_dim});
    for (int i=0; i<n; i++) {
        assert(tokens[i] < vocabulary && tokens[i] >= 0);
        std::copy(&weight_.data_[tokens[i] * latent_dim], &weight_.data_[tokens[i] * latent_dim + latent_dim], &res.data_[i * latent_dim]);
    }
    return res;
}
Tensor Embedding::forward(int token) const {
    Tensor res({latent_dim});
    assert(token < vocabulary && token >= 0);
    std::copy(&weight_.data_[token * latent_dim], &weight_.data_[token * latent_dim + latent_dim], &res.data_[0]);
    return res;
}
Tensor Embedding::inverse(const Tensor& x) {
    assert(x.shape_[x.shape_.size() - 1] == latent_dim);
    if (x.shape_.size() == 1) {
        Functions::matmul_inplace(weight_, x, workplace_);
        return workplace_;
    }
    else {
        return Functions::matmul(weight_, x);
    }
}

AttentionBlock::AttentionBlock(int max_len, int latent_dim, int m, float base_rope) : 
    Q_(latent_dim, latent_dim), K_(latent_dim, latent_dim), V_(latent_dim, latent_dim), W_(latent_dim, latent_dim),
    rope_(base_rope, max_len, latent_dim / m), KV(max_len, latent_dim / m, m),
    max_len(max_len), latent_dim(latent_dim), m(m) {
    // miao~
}
AttentionBlock::~AttentionBlock() {
    // miao~
}

void AttentionBlock::load(const std::string& base_path) {
    Q_.load(base_path + "/Q");
    K_.load(base_path + "/K");
    V_.load(base_path + "/V");
    W_.load(base_path + "/W");
}
Tensor AttentionBlock::forward(const Tensor& x, bool is_prefill) {
    assert(x.shape_[x.shape_.size() - 1] == latent_dim);
    std::vector<int> split_shape = x.shape_;
    split_shape[split_shape.size() - 1] = m;
    split_shape.push_back(latent_dim / m);
    int dim = latent_dim / m;
    Tensor Q = Q_.forward(x).reshape(split_shape);
    Tensor K = K_.forward(x).reshape(split_shape);
    Tensor V = V_.forward(x).reshape(split_shape);
    if (is_prefill) {
        rope_.forward(Q);
        rope_.forward(K);
        return W_.forward(MultiHeadAttention::prefill(Q, K, V, KV).reshape(x.shape_));
    }
    else {
        rope_.forward(Q, KV.n);
        rope_.forward(K, KV.n);
        return W_.forward(MultiHeadAttention::forward(Q, K, V, KV).reshape(x.shape_));
    }
}

TransformerBlock::TransformerBlock(int max_len, int latent_dim, int m, float base_rope, int hidden_dim) :
    attn_(max_len, latent_dim, m, base_rope), ffn_(latent_dim, hidden_dim), 
    norm1_(latent_dim), norm2_(latent_dim) {
    // miao~
}
TransformerBlock::~TransformerBlock() {
    // miao~
}

void TransformerBlock::load(const std::string& base_path) {
    attn_.load(base_path + "/attn");
    ffn_.load(base_path + "/ffn");
    norm1_.load(base_path + "/norm1");
    norm2_.load(base_path + "/norm2");
}

void TransformerBlock::forward(Tensor& x, bool is_prefill) {
    x.add(attn_.forward(norm1_.forward(x), is_prefill));
    x.add(ffn_.forward(norm2_.forward(x)));
}

Transformer::Transformer(int max_len, int latent_dim, int num_heads, float base_rope, 
        int hidden_dim, int vocabulary_size, int num_layers):
        embedding_(vocabulary_size, latent_dim), norm_(latent_dim){
    for (int i=0; i<num_layers; i++) {
        blocks_.push_back(TransformerBlock(max_len, latent_dim, num_heads, base_rope, hidden_dim));
    }
}
Transformer::~Transformer() {
    // miao~
}

void Transformer::load(const std::string& base_path) {
    embedding_.load(base_path + "/embedding");
    norm_.load(base_path + "/norm");
    for (int i=0; i<blocks_.size(); i++) {
        blocks_[i].load(base_path + "/blocks/" + std::to_string(i));
    }
}
Tensor Transformer::forward(const std::vector<int>& tokens) {
    Tensor x = embedding_.forward(tokens);
    for (auto& block : blocks_) {
        block.forward(x, true);
    }
    return embedding_.inverse(norm_.forward(x));
}
Tensor Transformer::forward(int token) {
    Tensor x = embedding_.forward(token);
    for (auto& block : blocks_) {
        block.forward(x, false);
    }
    return embedding_.inverse(norm_.forward(x));
}