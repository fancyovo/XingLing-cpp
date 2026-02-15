#include "tests.h"

static void init_attentionblock_weights(AttentionBlock& blk, std::mt19937& rng) {
    init_linear_weights(blk.Q_, rng);
    init_linear_weights(blk.K_, rng);
    init_linear_weights(blk.V_, rng);
    init_linear_weights(blk.W_, rng);
    blk.KV.n = 0;
}

static void copy_attentionblock_weights(const AttentionBlock& src, AttentionBlock& dst) {
    copy_linear_weights(src.Q_, dst.Q_);
    copy_linear_weights(src.K_, dst.K_);
    copy_linear_weights(src.V_, dst.V_);
    copy_linear_weights(src.W_, dst.W_);
    dst.KV.n = 0;
}

static void test_attentionblock_prefill() {
    int N=37;
    std::mt19937 rng(123);

    AttentionBlock cpu(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, "cpu");
    AttentionBlock gpu(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, "cuda");
    init_attentionblock_weights(cpu, rng);
    copy_attentionblock_weights(cpu, gpu);

    Tensor x_cpu({N, LATENT_DIM}, "cpu");
    Tensor x_gpu({N, LATENT_DIM}, "cuda");
    fill_random(x_cpu, rng, 0.1f);
    copy_tensor_data(x_cpu, x_gpu);

    Tensor out_cpu({N, LATENT_DIM}, "cpu");
    Tensor out_gpu({N, LATENT_DIM}, "cuda");

    cpu.forward(x_cpu, out_cpu, true);
    gpu.forward(x_gpu, out_gpu, true);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> out_h = copy_tensor_to_host(out_gpu);
    expect_allclose(out_h, copy_tensor_to_host(out_cpu),
                    2e-2f, 2e-2f, "AttentionBlock prefill cpu vs gpu");

    // KV cache compare
    int total = N * NUM_HEADS * HEAD_DIM;
    std::vector<float> k_cpu(cpu.KV.K_.h_ptr, cpu.KV.K_.h_ptr + total);
    std::vector<float> v_cpu(cpu.KV.V_.h_ptr, cpu.KV.V_.h_ptr + total);
    std::vector<float> k_gpu = copy_tensor_to_host(gpu.KV.K_);
    std::vector<float> v_gpu = copy_tensor_to_host(gpu.KV.V_);
    k_gpu.resize(total); v_gpu.resize(total);
    expect_allclose(k_gpu, k_cpu, 1e-3f, 1e-3f, "AttentionBlock KV.K prefill");
    expect_allclose(v_gpu, v_cpu, 1e-3f, 1e-3f, "AttentionBlock KV.V prefill");
}

static void test_attentionblock_decode() {
    int N=13;
    std::mt19937 rng(456);

    AttentionBlock cpu(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, "cpu");
    AttentionBlock gpu(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, "cuda");
    init_attentionblock_weights(cpu, rng);
    copy_attentionblock_weights(cpu, gpu);

    Tensor x0_cpu({N, LATENT_DIM}, "cpu");
    Tensor x0_gpu({N, LATENT_DIM}, "cuda");
    fill_random(x0_cpu, rng, 0.1f);
    copy_tensor_data(x0_cpu, x0_gpu);

    Tensor out0_cpu({N, LATENT_DIM}, "cpu");
    Tensor out0_gpu({N, LATENT_DIM}, "cuda");
    cpu.forward(x0_cpu, out0_cpu, true);
    gpu.forward(x0_gpu, out0_gpu, true);
    CHECK_CUDA(cudaDeviceSynchronize());

    Tensor x1_cpu({LATENT_DIM}, "cpu");
    Tensor x1_gpu({LATENT_DIM}, "cuda");
    fill_random(x1_cpu, rng, 0.1f);
    copy_tensor_data(x1_cpu, x1_gpu);

    Tensor out1_cpu({LATENT_DIM}, "cpu");
    Tensor out1_gpu({LATENT_DIM}, "cuda");
    cpu.forward(x1_cpu, out1_cpu, false);
    gpu.forward(x1_gpu, out1_gpu, false);
    CHECK_CUDA(cudaDeviceSynchronize());

    expect_allclose(copy_tensor_to_host(out1_gpu),
                    copy_tensor_to_host(out1_cpu),
                    2e-2f, 2e-2f, "AttentionBlock decode cpu vs gpu");

    int total = (N+1) * NUM_HEADS * HEAD_DIM;
    std::vector<float> k_cpu(cpu.KV.K_.h_ptr, cpu.KV.K_.h_ptr + total);
    std::vector<float> v_cpu(cpu.KV.V_.h_ptr, cpu.KV.V_.h_ptr + total);
    std::vector<float> k_gpu = copy_tensor_to_host(gpu.KV.K_);
    std::vector<float> v_gpu = copy_tensor_to_host(gpu.KV.V_);
    k_gpu.resize(total); v_gpu.resize(total);
    expect_allclose(k_gpu, k_cpu, 1e-3f, 1e-3f, "AttentionBlock KV.K decode");
    expect_allclose(v_gpu, v_cpu, 1e-3f, 1e-3f, "AttentionBlock KV.V decode");
}

static void speed_attentionblock_device(const std::string& device) {
    std::vector<int> lens={64,128,256,512};
    std::mt19937 rng(7);

    AttentionBlock blk(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, device);
    init_attentionblock_weights(blk, rng);

    for (int N: lens) {
        Tensor x({N, LATENT_DIM}, device);
        Tensor out({N, LATENT_DIM}, device);
        fill_random(x, rng, 0.1f);

        auto fn_prefill = [&](){ blk.forward(x, out, true); if(device=="cuda") cudaDeviceSynchronize(); };
        float t = (device=="cpu") ? time_cpu_ms(fn_prefill,1,2) : time_cuda_ms(fn_prefill,1,2);
        print_speed("AttentionBlock", device, "prefill", N, t);

        // decode 128
        Tensor v({LATENT_DIM}, device);
        Tensor vout({LATENT_DIM}, device);
        fill_random(v, rng, 0.1f);

        // prefill once for kv
        blk.forward(x, out, true);

        auto fn_decode = [&](){
            blk.KV.n = N;
            for(int i=0;i<128;i++) blk.forward(v, vout, false);
            if(device=="cuda") cudaDeviceSynchronize();
        };
        float t2 = (device=="cpu") ? time_cpu_ms(fn_decode,1,2) : time_cuda_ms(fn_decode,1,2);
        print_speed("AttentionBlock", device, "decode(128)", N, t2);
    }
}

void test_attentionblock(bool do_correct, bool do_speed) {
    if (do_correct) {
        test_attentionblock_prefill();
        test_attentionblock_decode();
    }
    if (do_speed) {
        speed_attentionblock_device("cpu");
        if (cuda_available()) speed_attentionblock_device("cuda");
    }
}
