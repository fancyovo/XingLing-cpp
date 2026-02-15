#include "tests.h"

static void init_transformerblock_weights(TransformerBlock& blk, std::mt19937& rng) {
    // attn
    init_linear_weights(blk.attn_.Q_, rng);
    init_linear_weights(blk.attn_.K_, rng);
    init_linear_weights(blk.attn_.V_, rng);
    init_linear_weights(blk.attn_.W_, rng);
    // ffn
    init_linear_weights(blk.ffn_.W1_, rng);
    init_linear_weights(blk.ffn_.W2_, rng);
    init_linear_weights(blk.ffn_.V_,  rng);
    // norms
    init_rmsnorm(blk.norm1_, rng);
    init_rmsnorm(blk.norm2_, rng);
    blk.attn_.KV.n = 0;
}

static void copy_transformerblock_weights(const TransformerBlock& src, TransformerBlock& dst) {
    copy_linear_weights(src.attn_.Q_, dst.attn_.Q_);
    copy_linear_weights(src.attn_.K_, dst.attn_.K_);
    copy_linear_weights(src.attn_.V_, dst.attn_.V_);
    copy_linear_weights(src.attn_.W_, dst.attn_.W_);
    copy_linear_weights(src.ffn_.W1_, dst.ffn_.W1_);
    copy_linear_weights(src.ffn_.W2_, dst.ffn_.W2_);
    copy_linear_weights(src.ffn_.V_,  dst.ffn_.V_);
    copy_rmsnorm(src.norm1_, dst.norm1_);
    copy_rmsnorm(src.norm2_, dst.norm2_);
    dst.attn_.KV.n = 0;
}

static void test_transformerblock_prefill() {
    int N=13;
    std::mt19937 rng(123);

    TransformerBlock cpu(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, FFN_DIM, "cpu");
    TransformerBlock gpu(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, FFN_DIM, "cuda");
    init_transformerblock_weights(cpu, rng);
    copy_transformerblock_weights(cpu, gpu);

    Tensor x_cpu({N, LATENT_DIM}, "cpu");
    Tensor x_gpu({N, LATENT_DIM}, "cuda");
    fill_random(x_cpu, rng, 0.1f);
    copy_tensor_data(x_cpu, x_gpu);

    cpu.forward(x_cpu, true);
    gpu.forward(x_gpu, true);
    CHECK_CUDA(cudaDeviceSynchronize());

    expect_allclose(copy_tensor_to_host(x_gpu),
                    copy_tensor_to_host(x_cpu),
                    3e-2f, 3e-2f,
                    "TransformerBlock prefill cpu vs gpu");

    int total = N * NUM_HEADS * HEAD_DIM;
    std::vector<float> k_cpu(cpu.attn_.KV.K_.h_ptr, cpu.attn_.KV.K_.h_ptr + total);
    std::vector<float> v_cpu(cpu.attn_.KV.V_.h_ptr, cpu.attn_.KV.V_.h_ptr + total);
    std::vector<float> k_gpu = copy_tensor_to_host(gpu.attn_.KV.K_);
    std::vector<float> v_gpu = copy_tensor_to_host(gpu.attn_.KV.V_);
    k_gpu.resize(total); v_gpu.resize(total);
    expect_allclose(k_gpu, k_cpu, 1e-3f, 1e-3f, "TransformerBlock KV.K prefill");
    expect_allclose(v_gpu, v_cpu, 1e-3f, 1e-3f, "TransformerBlock KV.V prefill");
}

static void test_transformerblock_decode() {
    int N=11;
    std::mt19937 rng(456);

    TransformerBlock cpu(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, FFN_DIM, "cpu");
    TransformerBlock gpu(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, FFN_DIM, "cuda");
    init_transformerblock_weights(cpu, rng);
    copy_transformerblock_weights(cpu, gpu);

    // prefill
    Tensor x0_cpu({N, LATENT_DIM}, "cpu");
    Tensor x0_gpu({N, LATENT_DIM}, "cuda");
    fill_random(x0_cpu, rng, 0.1f);
    copy_tensor_data(x0_cpu, x0_gpu);
    cpu.forward(x0_cpu, true);
    gpu.forward(x0_gpu, true);
    CHECK_CUDA(cudaDeviceSynchronize());

    // decode
    Tensor x1_cpu({LATENT_DIM}, "cpu");
    Tensor x1_gpu({LATENT_DIM}, "cuda");
    fill_random(x1_cpu, rng, 0.1f);
    copy_tensor_data(x1_cpu, x1_gpu);

    cpu.forward(x1_cpu, false);
    gpu.forward(x1_gpu, false);
    CHECK_CUDA(cudaDeviceSynchronize());

    expect_allclose(copy_tensor_to_host(x1_gpu),
                    copy_tensor_to_host(x1_cpu),
                    3e-2f, 3e-2f,
                    "TransformerBlock decode cpu vs gpu");

    int total = (N+1) * NUM_HEADS * HEAD_DIM;
    std::vector<float> k_cpu(cpu.attn_.KV.K_.h_ptr, cpu.attn_.KV.K_.h_ptr + total);
    std::vector<float> v_cpu(cpu.attn_.KV.V_.h_ptr, cpu.attn_.KV.V_.h_ptr + total);
    std::vector<float> k_gpu = copy_tensor_to_host(gpu.attn_.KV.K_);
    std::vector<float> v_gpu = copy_tensor_to_host(gpu.attn_.KV.V_);
    k_gpu.resize(total); v_gpu.resize(total);
    expect_allclose(k_gpu, k_cpu, 1e-3f, 1e-3f, "TransformerBlock KV.K decode");
    expect_allclose(v_gpu, v_cpu, 1e-3f, 1e-3f, "TransformerBlock KV.V decode");
}

static void speed_transformerblock_device(const std::string& device) {
    std::vector<int> lens={64,128,256,512};
    std::mt19937 rng(7);

    TransformerBlock blk(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, FFN_DIM, device);
    init_transformerblock_weights(blk, rng);

    for (int N: lens) {
        Tensor x({N, LATENT_DIM}, device);
        fill_random(x, rng, 0.1f);

        auto fn_prefill = [&](){ blk.forward(x, true); if(device=="cuda") cudaDeviceSynchronize(); };
        float t = (device=="cpu") ? time_cpu_ms(fn_prefill,1,1) : time_cuda_ms(fn_prefill,1,1);
        print_speed("TransformerBlock", device, "prefill", N, t);

        // decode
        Tensor v({LATENT_DIM}, device);
        fill_random(v, rng, 0.1f);
        blk.forward(x, true); // prefill to init KV

        auto fn_decode = [&](){
            blk.attn_.KV.n = N;
            for(int i=0;i<128;i++) blk.forward(v, false);
            if(device=="cuda") cudaDeviceSynchronize();
        };
        float t2 = (device=="cpu") ? time_cpu_ms(fn_decode,1,1) : time_cuda_ms(fn_decode,1,1);
        print_speed("TransformerBlock", device, "decode(128)", N, t2);
    }
}

void test_transformerblock(bool do_correct, bool do_speed) {
    if (do_correct) {
        test_transformerblock_prefill();
        test_transformerblock_decode();
    }
    if (do_speed) {
        speed_transformerblock_device("cpu");
        if (cuda_available()) speed_transformerblock_device("cuda");
    }
}
