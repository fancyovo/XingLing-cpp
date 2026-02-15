#include "tests.h"
#include <cmath>

static void naive_attn_prefill(
    const float* Q, const float* K, const float* V,
    float* O, float* KV_K, float* KV_V,
    int N, int m, int C
){
    float scale = 1.0f / std::sqrt((float)C);
    for (int t=0;t<N;t++){
        for (int h=0;h<m;h++){
            // scores
            std::vector<float> scores(t+1);
            float maxv=-1e30f;
            for (int j=0;j<=t;j++){
                float dot=0.f;
                for (int c=0;c<C;c++){
                    dot += Q[(t*m+h)*C + c] * K[(j*m+h)*C + c];
                }
                dot *= scale;
                scores[j] = dot;
                if (dot > maxv) maxv = dot;
            }
            float sum=0.f;
            for (int j=0;j<=t;j++) sum += std::exp(scores[j]-maxv);
            for (int c=0;c<C;c++){
                float out=0.f;
                for (int j=0;j<=t;j++){
                    float w = std::exp(scores[j]-maxv) / sum;
                    out += w * V[(j*m+h)*C + c];
                }
                O[(t*m+h)*C + c] = out;
            }
        }
    }
    std::memcpy(KV_K, K, N*m*C*sizeof(float));
    std::memcpy(KV_V, V, N*m*C*sizeof(float));
}

static void naive_attn_decode(
    const float* Q, const float* K, const float* V,
    float* O, float* KV_K, float* KV_V,
    int& n, int m, int C
){
    // append new KV
    for (int h=0;h<m;h++){
        for (int c=0;c<C;c++){
            KV_K[(n*m+h)*C + c] = K[h*C + c];
            KV_V[(n*m+h)*C + c] = V[h*C + c];
        }
    }
    float scale = 1.0f / std::sqrt((float)C);
    int N = n; // old length
    for (int h=0;h<m;h++){
        std::vector<float> scores(N+1);
        float maxv=-1e30f;
        for (int j=0;j<=N;j++){
            float dot=0.f;
            for (int c=0;c<C;c++){
                dot += Q[h*C + c] * KV_K[(j*m+h)*C + c];
            }
            dot *= scale;
            scores[j]=dot;
            if (dot>maxv) maxv=dot;
        }
        float sum=0.f;
        for (int j=0;j<=N;j++) sum += std::exp(scores[j]-maxv);
        for (int c=0;c<C;c++){
            float out=0.f;
            for (int j=0;j<=N;j++){
                float w = std::exp(scores[j]-maxv)/sum;
                out += w * KV_V[(j*m+h)*C + c];
            }
            O[h*C + c] = out;
        }
    }
    n++;
}

static void test_attn_prefill_device(const std::string& device) {
    int N=37, m=NUM_HEADS, C=HEAD_DIM;
    std::mt19937 rng(123);

    Tensor Q({N,m,C}, device);
    Tensor K({N,m,C}, device);
    Tensor V({N,m,C}, device);
    Tensor O({N,m,C}, device);
    fill_random(Q, rng, 0.1f);
    fill_random(K, rng, 0.1f);
    fill_random(V, rng, 0.1f);

    KVcache KV(MAX_SEQ_LEN, C, m, device);

    // reference
    std::vector<float> Qh = copy_tensor_to_host(Q);
    std::vector<float> Kh = copy_tensor_to_host(K);
    std::vector<float> Vh = copy_tensor_to_host(V);

    std::vector<float> refO(N*m*C);
    std::vector<float> refK(MAX_SEQ_LEN*m*C, 0.f);
    std::vector<float> refV(MAX_SEQ_LEN*m*C, 0.f);

    naive_attn_prefill(Qh.data(), Kh.data(), Vh.data(),
                       refO.data(), refK.data(), refV.data(),
                       N, m, C);

    MultiHeadAttention::prefill(Q,K,V,O,KV);
    if (device=="cuda") CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> Oh = copy_tensor_to_host(O);
    std::vector<float> KVKh = copy_tensor_to_host(KV.K_);
    std::vector<float> KVVh = copy_tensor_to_host(KV.V_);

    expect_allclose(Oh, refO, 2e-2f, 2e-2f, "Attn prefill output " + device);

    // KV cache check (只比较前 N)
    std::vector<float> refKcut(refK.begin(), refK.begin()+N*m*C);
    std::vector<float> refVcut(refV.begin(), refV.begin()+N*m*C);
    std::vector<float> Kcut(KVKh.begin(), KVKh.begin()+N*m*C);
    std::vector<float> Vcut(KVVh.begin(), KVVh.begin()+N*m*C);
    expect_allclose(Kcut, refKcut, 1e-3f, 1e-3f, "Attn prefill KV.K " + device);
    expect_allclose(Vcut, refVcut, 1e-3f, 1e-3f, "Attn prefill KV.V " + device);
}

static void test_attn_decode_device(const std::string& device) {
    int N=13, m=NUM_HEADS, C=HEAD_DIM;
    std::mt19937 rng(456);

    Tensor Q0({N,m,C}, device);
    Tensor K0({N,m,C}, device);
    Tensor V0({N,m,C}, device);
    Tensor O0({N,m,C}, device);
    fill_random(Q0, rng, 0.1f);
    fill_random(K0, rng, 0.1f);
    fill_random(V0, rng, 0.1f);

    KVcache KV(MAX_SEQ_LEN, C, m, device);

    // prefill first
    MultiHeadAttention::prefill(Q0, K0, V0, O0, KV);
    if (device=="cuda") CHECK_CUDA(cudaDeviceSynchronize());

    // decode input (single token)
    Tensor Q1({m,C}, device);
    Tensor K1({m,C}, device);
    Tensor V1({m,C}, device);
    Tensor O1({m,C}, device);
    fill_random(Q1, rng, 0.1f);
    fill_random(K1, rng, 0.1f);
    fill_random(V1, rng, 0.1f);

    // reference KV from CPU
    std::vector<float> KVKh = copy_tensor_to_host(KV.K_);
    std::vector<float> KVVh = copy_tensor_to_host(KV.V_);
    int n_ref = N;

    std::vector<float> Q1h = copy_tensor_to_host(Q1);
    std::vector<float> K1h = copy_tensor_to_host(K1);
    std::vector<float> V1h = copy_tensor_to_host(V1);

    std::vector<float> refO(m*C);
    naive_attn_decode(Q1h.data(), K1h.data(), V1h.data(),
                      refO.data(), KVKh.data(), KVVh.data(),
                      n_ref, m, C);

    MultiHeadAttention::forward(Q1, K1, V1, O1, KV);
    if (device=="cuda") CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> Oh = copy_tensor_to_host(O1);
    expect_allclose(Oh, refO, 2e-2f, 2e-2f, "Attn decode output " + device);

    // KV cache check (前 n_ref = N+1)
    std::vector<float> KVKh2 = copy_tensor_to_host(KV.K_);
    std::vector<float> KVVh2 = copy_tensor_to_host(KV.V_);
    int total = n_ref * m * C;
    std::vector<float> refK(KVKh.begin(), KVKh.begin()+total);
    std::vector<float> refV(KVVh.begin(), KVVh.begin()+total);
    std::vector<float> outK(KVKh2.begin(), KVKh2.begin()+total);
    std::vector<float> outV(KVVh2.begin(), KVVh2.begin()+total);
    expect_allclose(outK, refK, 1e-3f, 1e-3f, "Attn decode KV.K " + device);
    expect_allclose(outV, refV, 1e-3f, 1e-3f, "Attn decode KV.V " + device);
}

static void test_attn_kv_reset_logic() {
    std::mt19937 rng(777);
    int m=NUM_HEADS, C=HEAD_DIM;

    for (int round=0; round<3; round++) {
        int N = 5 + round*3;
        Tensor Q({N,m,C}, "cpu");
        Tensor K({N,m,C}, "cpu");
        Tensor V({N,m,C}, "cpu");
        Tensor O({N,m,C}, "cpu");
        fill_random(Q, rng, 0.1f);
        fill_random(K, rng, 0.1f);
        fill_random(V, rng, 0.1f);

        KVcache KV(MAX_SEQ_LEN, C, m, "cpu");
        MultiHeadAttention::prefill(Q,K,V,O,KV);
        expect_true(KV.n==N, "KV reset round " + std::to_string(round) + " (prefill)");

        Tensor Q1({m,C}, "cpu");
        Tensor K1({m,C}, "cpu");
        Tensor V1({m,C}, "cpu");
        Tensor O1({m,C}, "cpu");
        fill_random(Q1, rng, 0.1f);
        fill_random(K1, rng, 0.1f);
        fill_random(V1, rng, 0.1f);
        MultiHeadAttention::forward(Q1,K1,V1,O1,KV);
        expect_true(KV.n==N+1, "KV append round " + std::to_string(round) + " (decode)");
    }
}

static void speed_attn_device(const std::string& device) {
    std::vector<int> lens={64,128,256,512};
    std::mt19937 rng(7);

    for (int N: lens) {
        Tensor Q({N,NUM_HEADS,HEAD_DIM}, device);
        Tensor K({N,NUM_HEADS,HEAD_DIM}, device);
        Tensor V({N,NUM_HEADS,HEAD_DIM}, device);
        Tensor O({N,NUM_HEADS,HEAD_DIM}, device);
        fill_random(Q, rng, 0.1f);
        fill_random(K, rng, 0.1f);
        fill_random(V, rng, 0.1f);

        KVcache KV(MAX_SEQ_LEN, HEAD_DIM, NUM_HEADS, device);

        auto fn_prefill = [&](){ MultiHeadAttention::prefill(Q,K,V,O,KV); if(device=="cuda") cudaDeviceSynchronize(); };
        float t = (device=="cpu") ? time_cpu_ms(fn_prefill,1,2) : time_cuda_ms(fn_prefill,1,3);
        print_speed("Attn", device, "prefill", N, t);

        // decode 128 steps
        Tensor Q1({NUM_HEADS,HEAD_DIM}, device);
        Tensor K1({NUM_HEADS,HEAD_DIM}, device);
        Tensor V1({NUM_HEADS,HEAD_DIM}, device);
        Tensor O1({NUM_HEADS,HEAD_DIM}, device);
        fill_random(Q1, rng, 0.1f);
        fill_random(K1, rng, 0.1f);
        fill_random(V1, rng, 0.1f);

        // prefill once to init KV
        MultiHeadAttention::prefill(Q,K,V,O,KV);

        auto fn_decode = [&](){
            KV.n = N;
            for(int i=0;i<128;i++) MultiHeadAttention::forward(Q1,K1,V1,O1,KV);
            if(device=="cuda") cudaDeviceSynchronize();
        };
        float t2 = (device=="cpu") ? time_cpu_ms(fn_decode,1,2) : time_cuda_ms(fn_decode,1,2);
        print_speed("Attn", device, "decode(128)", N, t2);
    }
}

void test_attn(bool do_correct, bool do_speed) {
    if (do_correct) {
        test_attn_prefill_device("cpu");
        test_attn_decode_device("cpu");
        if (cuda_available()) {
            test_attn_prefill_device("cuda");
            test_attn_decode_device("cuda");
        } else test_skip("Attn CUDA tests (no GPU)");
        test_attn_kv_reset_logic(); // 3 次确认
    }
    if (do_speed) {
        speed_attn_device("cpu");
        if (cuda_available()) speed_attn_device("cuda");
    }
}
