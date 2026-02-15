#include "tests.h"

static void init_swishglu_weights(SwishGLU& ffn, std::mt19937& rng) {
    init_linear_weights(ffn.W1_, rng);
    init_linear_weights(ffn.W2_, rng);
    init_linear_weights(ffn.V_, rng);
}

static void copy_swishglu_weights(const SwishGLU& src, SwishGLU& dst) {
    copy_linear_weights(src.W1_, dst.W1_);
    copy_linear_weights(src.W2_, dst.W2_);
    copy_linear_weights(src.V_,  dst.V_);
}

static void test_swishglu_prefill() {
    int N=37;
    std::mt19937 rng(123);

    SwishGLU cpu(LATENT_DIM, FFN_DIM, "cpu");
    SwishGLU gpu(LATENT_DIM, FFN_DIM, "cuda");
    init_swishglu_weights(cpu, rng);
    copy_swishglu_weights(cpu, gpu);

    Tensor x_cpu({N, LATENT_DIM}, "cpu");
    Tensor x_gpu({N, LATENT_DIM}, "cuda");
    fill_random(x_cpu, rng, 0.1f);
    copy_tensor_data(x_cpu, x_gpu);

    cpu.forward(x_cpu);
    gpu.forward(x_gpu);
    CHECK_CUDA(cudaDeviceSynchronize());

    expect_allclose(copy_tensor_to_host(x_gpu),
                    copy_tensor_to_host(x_cpu),
                    2e-2f, 2e-2f, "SwishGLU prefill cpu vs gpu");
}

static void test_swishglu_decode() {
    std::mt19937 rng(456);

    SwishGLU cpu(LATENT_DIM, FFN_DIM, "cpu");
    SwishGLU gpu(LATENT_DIM, FFN_DIM, "cuda");
    init_swishglu_weights(cpu, rng);
    copy_swishglu_weights(cpu, gpu);

    Tensor x_cpu({LATENT_DIM}, "cpu");
    Tensor x_gpu({LATENT_DIM}, "cuda");
    fill_random(x_cpu, rng, 0.1f);
    copy_tensor_data(x_cpu, x_gpu);

    cpu.forward_vec(x_cpu);
    gpu.forward_vec(x_gpu);
    CHECK_CUDA(cudaDeviceSynchronize());

    expect_allclose(copy_tensor_to_host(x_gpu),
                    copy_tensor_to_host(x_cpu),
                    2e-2f, 2e-2f, "SwishGLU decode cpu vs gpu");
}

static void speed_swishglu_device(const std::string& device) {
    std::vector<int> lens={64,128,256,512};
    std::mt19937 rng(7);

    SwishGLU ffn(LATENT_DIM, FFN_DIM, device);
    init_swishglu_weights(ffn, rng);

    for (int N: lens) {
        Tensor x({N, LATENT_DIM}, device);
        fill_random(x, rng, 0.1f);

        auto fn_prefill = [&](){ ffn.forward(x); if(device=="cuda") cudaDeviceSynchronize(); };
        float t = (device=="cpu") ? time_cpu_ms(fn_prefill,1,2) : time_cuda_ms(fn_prefill,1,2);
        print_speed("SwishGLU", device, "prefill", N, t);

        Tensor v({LATENT_DIM}, device);
        fill_random(v, rng, 0.1f);
        auto fn_decode = [&](){ for(int i=0;i<128;i++) ffn.forward_vec(v); if(device=="cuda") cudaDeviceSynchronize(); };
        float t2 = (device=="cpu") ? time_cpu_ms(fn_decode,1,2) : time_cuda_ms(fn_decode,1,2);
        print_speed("SwishGLU", device, "decode(128)", N, t2);
    }
}

void test_swishglu(bool do_correct, bool do_speed) {
    if (do_correct) {
        test_swishglu_prefill();
        test_swishglu_decode();
    }
    if (do_speed) {
        speed_swishglu_device("cpu");
        if (cuda_available()) speed_swishglu_device("cuda");
    }
}
