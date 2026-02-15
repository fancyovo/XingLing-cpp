#include "tests.h"

// ====== 随机初始化 Transformer ======
static void init_transformer_weights(Transformer& tr, std::mt19937& rng) {
    init_embedding(tr.embedding_, rng, 0.02f);
    init_rmsnorm(tr.norm_, rng);
    for (auto& blk : tr.blocks_) {
        init_linear_weights(blk->attn_.Q_, rng);
        init_linear_weights(blk->attn_.K_, rng);
        init_linear_weights(blk->attn_.V_, rng);
        init_linear_weights(blk->attn_.W_, rng);
        init_linear_weights(blk->ffn_.W1_, rng);
        init_linear_weights(blk->ffn_.W2_, rng);
        init_linear_weights(blk->ffn_.V_,  rng);
        init_rmsnorm(blk->norm1_, rng);
        init_rmsnorm(blk->norm2_, rng);
        blk->attn_.KV.n = 0;
    }
}

static void copy_transformer_weights(const Transformer& src, Transformer& dst) {
    copy_embedding(src.embedding_, dst.embedding_);
    copy_rmsnorm(src.norm_, dst.norm_);
    for (int i=0;i<src.blocks_.size();i++){
        auto& s = src.blocks_[i];
        auto& d = dst.blocks_[i];
        copy_linear_weights(s->attn_.Q_, d->attn_.Q_);
        copy_linear_weights(s->attn_.K_, d->attn_.K_);
        copy_linear_weights(s->attn_.V_, d->attn_.V_);
        copy_linear_weights(s->attn_.W_, d->attn_.W_);
        copy_linear_weights(s->ffn_.W1_, d->ffn_.W1_);
        copy_linear_weights(s->ffn_.W2_, d->ffn_.W2_);
        copy_linear_weights(s->ffn_.V_,  d->ffn_.V_);
        copy_rmsnorm(s->norm1_, d->norm1_);
        copy_rmsnorm(s->norm2_, d->norm2_);
        d->attn_.KV.n = 0;
    }
}

static void test_transformer_prefill_decode() {
    int N=17;
    std::mt19937 rng(123);

    Transformer cpu(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, FFN_DIM,
                    VOCAB_SIZE, NUM_LAYERS, "cpu");
    Transformer gpu(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, FFN_DIM,
                    VOCAB_SIZE, NUM_LAYERS, "cuda");

    init_transformer_weights(cpu, rng);
    copy_transformer_weights(cpu, gpu);

    std::vector<int> tokens(N);
    fill_random_int(tokens, 0, VOCAB_SIZE-1, rng);

    Tensor logits_cpu({VOCAB_SIZE}, "cpu");
    Tensor logits_gpu({VOCAB_SIZE}, "cuda");

    // prefill
    cpu.forward(tokens, logits_cpu);
    gpu.forward(tokens, logits_gpu);
    CHECK_CUDA(cudaDeviceSynchronize());

    expect_allclose(copy_tensor_to_host(logits_gpu),
                    copy_tensor_to_host(logits_cpu),
                    3e-2f, 3e-2f, "Transformer prefill cpu vs gpu");

    // decode
    int tok = tokens[0];
    cpu.forward(tok, logits_cpu);
    gpu.forward(tok, logits_gpu);
    CHECK_CUDA(cudaDeviceSynchronize());

    expect_allclose(copy_tensor_to_host(logits_gpu),
                    copy_tensor_to_host(logits_cpu),
                    3e-2f, 3e-2f, "Transformer decode cpu vs gpu");

    // KV cache对齐（只比较前 N+1）
    for (int i=0;i<NUM_LAYERS;i++){
        int total = (N+1) * NUM_HEADS * HEAD_DIM;
        auto& blk_cpu = cpu.blocks_[i];
        auto& blk_gpu = gpu.blocks_[i];

        std::vector<float> k_cpu(blk_cpu->attn_.KV.K_.h_ptr,
                                 blk_cpu->attn_.KV.K_.h_ptr + total);
        std::vector<float> v_cpu(blk_cpu->attn_.KV.V_.h_ptr,
                                 blk_cpu->attn_.KV.V_.h_ptr + total);

        std::vector<float> k_gpu = copy_tensor_to_host(blk_gpu->attn_.KV.K_);
        std::vector<float> v_gpu = copy_tensor_to_host(blk_gpu->attn_.KV.V_);
        k_gpu.resize(total); v_gpu.resize(total);

        expect_allclose(k_gpu, k_cpu, 1e-3f, 1e-3f,
                        "Transformer KV.K layer " + std::to_string(i));
        expect_allclose(v_gpu, v_cpu, 1e-3f, 1e-3f,
                        "Transformer KV.V layer " + std::to_string(i));
    }
}

// KV reset/append 逻辑重复 3 次确认
static void test_transformer_kv_logic() {
    std::mt19937 rng(777);

    Transformer tr(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, FFN_DIM,
                   VOCAB_SIZE, NUM_LAYERS, "cpu");
    init_transformer_weights(tr, rng);

    for (int round=0; round<3; round++) {
        int N = 5 + round*4;
        std::vector<int> tokens(N);
        fill_random_int(tokens, 0, VOCAB_SIZE-1, rng);

        Tensor logits({VOCAB_SIZE}, "cpu");
        tr.forward(tokens, logits);
        for (int i=0;i<NUM_LAYERS;i++){
            expect_true(tr.blocks_[i]->attn_.KV.n == N,
                        "KV reset round " + std::to_string(round) + " layer " + std::to_string(i));
        }

        // decode 2 steps
        for (int s=0;s<2;s++){
            int tok = tokens[s % tokens.size()];
            tr.forward(tok, logits);
        }
        for (int i=0;i<NUM_LAYERS;i++){
            expect_true(tr.blocks_[i]->attn_.KV.n == N+2,
                        "KV append round " + std::to_string(round) + " layer " + std::to_string(i));
        }
    }
}

static void test_weight_loading() {
    try {
        Transformer tr(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, FFN_DIM,
                       VOCAB_SIZE, NUM_LAYERS, "cpu");
        tr.load("../data/model"); // build/ 目录下运行
        std::vector<int> tokens = {1,2,3,4};
        Tensor logits({VOCAB_SIZE}, "cpu");
        tr.forward(tokens, logits);
        // 简单检查是否 NaN
        bool ok = true;
        for (int i=0;i<10;i++){
            if (std::isnan(logits.h_ptr[i]) || std::isinf(logits.h_ptr[i])) ok=false;
        }
        expect_true(ok, "Weight loading + forward sanity");
    } catch (std::exception& e) {
        test_fail(std::string("Weight loading failed: ") + e.what());
    }
}

static void speed_transformer_device(const std::string& device) {
    std::vector<int> lens={64,128,256,512};
    std::mt19937 rng(7);

    Transformer tr(MAX_SEQ_LEN, LATENT_DIM, NUM_HEADS, ROPE_BASE, FFN_DIM,
                   VOCAB_SIZE, NUM_LAYERS, device);
    init_transformer_weights(tr, rng);

    for (int N: lens) {
        std::vector<int> tokens(N);
        fill_random_int(tokens, 0, VOCAB_SIZE-1, rng);

        Tensor logits({VOCAB_SIZE}, device);

        auto fn_prefill = [&](){ tr.forward(tokens, logits); if(device=="cuda") cudaDeviceSynchronize(); };
        float t = (device=="cpu") ? time_cpu_ms(fn_prefill,1,1) : time_cuda_ms(fn_prefill,1,1);
        print_speed("Transformer", device, "prefill", N, t);

        // decode 128
        tr.forward(tokens, logits);
        auto fn_decode = [&](){
            for (auto& blk: tr.blocks_) blk->attn_.KV.n = N;
            int tok = tokens[0];
            for(int i=0;i<128;i++) tr.forward(tok, logits);
            if(device=="cuda") cudaDeviceSynchronize();
        };
        float t2 = (device=="cpu") ? time_cpu_ms(fn_decode,1,1) : time_cuda_ms(fn_decode,1,1);
        print_speed("Transformer", device, "decode(128)", N, t2);
    }
}

void test_transformer(bool do_correct, bool do_speed, bool do_weight) {
    if (do_correct) {
        test_transformer_prefill_decode();
        test_transformer_kv_logic();
    }
    if (do_speed) {
        speed_transformer_device("cpu");
        if (cuda_available()) speed_transformer_device("cuda");
    }
    if (do_weight) {
        test_weight_loading();
    }
}
