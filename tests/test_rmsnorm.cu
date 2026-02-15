#include "tests.h"
#include <cmath>

static void naive_rmsnorm(const float* input, const float* gamma,
                          float* output, int N, int K) {
    for (int i=0;i<N;i++){
        float sum=0.f;
        for (int j=0;j<K;j++){
            float v=input[i*K+j];
            sum += v*v;
        }
        float inv = 1.0f / std::sqrt(sum / K + 1e-6f);
        for (int j=0;j<K;j++){
            output[i*K+j] = input[i*K+j] * gamma[j] * inv;
        }
    }
}

static void test_rmsnorm_prefill_device(const std::string& device) {
    int N=37, K=LATENT_DIM;
    std::mt19937 rng(123);

    RMSNorm norm(K, device);
    init_rmsnorm(norm, rng);

    Tensor x({N,K}, device);
    fill_random(x, rng, 0.1f);

    // 备份输入
    std::vector<float> x_h = copy_tensor_to_host(x);

    norm.forward(x);
    if (device=="cuda") CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> ref(N*K);
    naive_rmsnorm(x_h.data(), norm.gamma_.h_ptr, ref.data(), N, K);

    std::vector<float> out_h = copy_tensor_to_host(x);
    expect_allclose(out_h, ref, 1e-3f, 1e-3f,
                    "RMSNorm prefill " + device);
}

static void test_rmsnorm_decode_device(const std::string& device) {
    int K=LATENT_DIM;
    std::mt19937 rng(456);

    RMSNorm norm(K, device);
    init_rmsnorm(norm, rng);

    Tensor x({K}, device);
    fill_random(x, rng, 0.1f);

    std::vector<float> x_h = copy_tensor_to_host(x);

    norm.forward(x);
    if (device=="cuda") CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> ref(K);
    naive_rmsnorm(x_h.data(), norm.gamma_.h_ptr, ref.data(), 1, K);

    std::vector<float> out_h = copy_tensor_to_host(x);
    expect_allclose(out_h, ref, 1e-3f, 1e-3f,
                    "RMSNorm decode " + device);
}

static void speed_rmsnorm_device(const std::string& device) {
    std::vector<int> lens = {64,128,256,512};
    std::mt19937 rng(7);

    RMSNorm norm(LATENT_DIM, device);
    init_rmsnorm(norm, rng);

    for (int N: lens) {
        Tensor x({N, LATENT_DIM}, device);
        fill_random(x, rng, 0.1f);

        auto fn_prefill = [&](){ norm.forward(x); if(device=="cuda") cudaDeviceSynchronize(); };
        float t = (device=="cpu") ? time_cpu_ms(fn_prefill,1,3) : time_cuda_ms(fn_prefill,1,5);
        print_speed("RMSNorm", device, "prefill", N, t);

        Tensor v({LATENT_DIM}, device);
        fill_random(v, rng, 0.1f);
        auto fn_decode = [&](){ for(int i=0;i<128;i++) norm.forward(v); if(device=="cuda") cudaDeviceSynchronize(); };
        float t2 = (device=="cpu") ? time_cpu_ms(fn_decode,1,3) : time_cuda_ms(fn_decode,1,3);
        print_speed("RMSNorm", device, "decode(128)", N, t2);
    }
}

void test_rmsnorm(bool do_correct, bool do_speed) {
    if (do_correct) {
        test_rmsnorm_prefill_device("cpu");
        test_rmsnorm_decode_device("cpu");
        if (cuda_available()) {
            test_rmsnorm_prefill_device("cuda");
            test_rmsnorm_decode_device("cuda");
        } else test_skip("RMSNorm CUDA tests (no GPU)");
    }
    if (do_speed) {
        speed_rmsnorm_device("cpu");
        if (cuda_available()) speed_rmsnorm_device("cuda");
    }
}
