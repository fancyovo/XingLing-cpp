#include "tests.h"
#include <cmath>

static void naive_embedding(
    const float* weight, const std::vector<int>& ids,
    float* output, int N, int K
){
    for (int i=0;i<N;i++){
        std::memcpy(output + i*K, weight + ids[i]*K, K*sizeof(float));
    }
}

static void naive_embedding_inv(
    const float* weight, const float* input,
    float* output, int V, int K
){
    for (int i=0;i<V;i++){
        float sum=0.f;
        for (int j=0;j<K;j++){
            sum += input[j] * weight[i*K + j];
        }
        output[i] = sum;
    }
}

static void test_embedding_prefill_device(const std::string& device) {
    int N=37, K=LATENT_DIM;
    std::mt19937 rng(123);

    Embedding emb(VOCAB_SIZE, K, device);
    init_embedding(emb, rng, 0.02f);

    std::vector<int> ids(N);
    fill_random_int(ids, 0, VOCAB_SIZE-1, rng);

    Tensor out({N,K}, device);
    emb.forward(ids, out);
    if (device=="cuda") CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> ref(N*K);
    naive_embedding(emb.weight_.h_ptr, ids, ref.data(), N, K);

    std::vector<float> out_h = copy_tensor_to_host(out);
    expect_allclose(out_h, ref, 1e-3f, 1e-3f,
                    "Embedding prefill " + device);
}

static void test_embedding_decode_device(const std::string& device) {
    int K=LATENT_DIM;
    std::mt19937 rng(456);

    Embedding emb(VOCAB_SIZE, K, device);
    init_embedding(emb, rng, 0.02f);

    int token = VOCAB_SIZE/2;
    Tensor out({K}, device);

    emb.forward(token, out);
    if (device=="cuda") CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> ref(K);
    std::memcpy(ref.data(), emb.weight_.h_ptr + token*K, K*sizeof(float));

    std::vector<float> out_h = copy_tensor_to_host(out);
    expect_allclose(out_h, ref, 1e-3f, 1e-3f,
                    "Embedding decode " + device);
}

static void test_embedding_inverse_device(const std::string& device) {
    int K=LATENT_DIM;
    std::mt19937 rng(789);

    Embedding emb(VOCAB_SIZE, K, device);
    init_embedding(emb, rng, 0.02f);

    Tensor x({K}, device);
    fill_random(x, rng, 0.1f);
    Tensor out({VOCAB_SIZE}, device);

    emb.inverse(x, out);
    if (device=="cuda") CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> ref(VOCAB_SIZE);
    std::vector<float> x_h = copy_tensor_to_host(x);
    naive_embedding_inv(emb.weight_.h_ptr, x_h.data(), ref.data(), VOCAB_SIZE, K);

    std::vector<float> out_h = copy_tensor_to_host(out);
    expect_allclose(out_h, ref, 1e-2f, 1e-2f,
                    "Embedding inverse " + device);
}

static void speed_embedding_device(const std::string& device) {
    std::vector<int> lens = {64,128,256,512};
    std::mt19937 rng(7);

    Embedding emb(VOCAB_SIZE, LATENT_DIM, device);
    init_embedding(emb, rng, 0.02f);

    for (int N: lens) {
        std::vector<int> ids(N);
        fill_random_int(ids, 0, VOCAB_SIZE-1, rng);

        Tensor out({N, LATENT_DIM}, device);
        auto fn_prefill = [&](){ emb.forward(ids, out); if(device=="cuda") cudaDeviceSynchronize(); };
        float t = (device=="cpu") ? time_cpu_ms(fn_prefill,1,3) : time_cuda_ms(fn_prefill,1,3);
        print_speed("Embedding", device, "prefill", N, t);

        Tensor vout({LATENT_DIM}, device);
        int token = ids[0];
        auto fn_decode = [&](){ for(int i=0;i<128;i++) emb.forward(token, vout); if(device=="cuda") cudaDeviceSynchronize(); };
        float t2 = (device=="cpu") ? time_cpu_ms(fn_decode,1,3) : time_cuda_ms(fn_decode,1,3);
        print_speed("Embedding", device, "decode(128)", N, t2);
    }
}

void test_embedding(bool do_correct, bool do_speed) {
    if (do_correct) {
        test_embedding_prefill_device("cpu");
        test_embedding_decode_device("cpu");
        test_embedding_inverse_device("cpu");
        if (cuda_available()) {
            test_embedding_prefill_device("cuda");
            test_embedding_decode_device("cuda");
            test_embedding_inverse_device("cuda");
        } else test_skip("Embedding CUDA tests (no GPU)");
    }
    if (do_speed) {
        speed_embedding_device("cpu");
        if (cuda_available()) speed_embedding_device("cuda");
    }
}
