#include "tests.h"
#include <cmath>

static void naive_rope(float* input, float base, int N, int m, int C, int start_pos=0) {
    for (int i=0;i<N;i++){
        int pos = i + start_pos;
        for (int j=0;j<m;j++){
            for (int k=0;k<C/2;k++){
                float theta = std::pow(base, -2.0f * k / C) * pos;
                float cosP = std::cos(theta);
                float sinP = std::sin(theta);
                int idx = i*m*C + j*C + k;
                float i0 = input[idx];
                float i1 = input[idx + C/2];
                float o0 =  i0 * cosP + i1 * sinP;
                float o1 = -i0 * sinP + i1 * cosP;
                input[idx] = o0;
                input[idx + C/2] = o1;
            }
        }
    }
}

static void test_rope_prefill_device(const std::string& device) {
    int N=37, m=NUM_HEADS, C=HEAD_DIM;
    std::mt19937 rng(123);

    Tensor x({N,m,C}, device);
    fill_random(x, rng, 0.1f);

    std::vector<float> ref = copy_tensor_to_host(x);
    naive_rope(ref.data(), ROPE_BASE, N, m, C, 0);

    RoPE::forward(x, ROPE_BASE);
    if (device=="cuda") CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> out_h = copy_tensor_to_host(x);
    expect_allclose(out_h, ref, 1e-3f, 1e-3f,
                    "RoPE prefill " + device);
}

static void test_rope_decode_device(const std::string& device) {
    int m=NUM_HEADS, C=HEAD_DIM, pos=13;
    std::mt19937 rng(456);

    Tensor x({m,C}, device);
    fill_random(x, rng, 0.1f);

    std::vector<float> ref = copy_tensor_to_host(x);
    naive_rope(ref.data(), ROPE_BASE, 1, m, C, pos);

    RoPE::forward_vec(x, pos, ROPE_BASE);
    if (device=="cuda") CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> out_h = copy_tensor_to_host(x);
    expect_allclose(out_h, ref, 1e-3f, 1e-3f,
                    "RoPE decode " + device);
}

static void speed_rope_device(const std::string& device) {
    std::vector<int> lens = {64,128,256,512};
    std::mt19937 rng(7);

    for (int N: lens) {
        Tensor x({N, NUM_HEADS, HEAD_DIM}, device);
        fill_random(x, rng, 0.1f);

        auto fn_prefill = [&](){ RoPE::forward(x, ROPE_BASE); if(device=="cuda") cudaDeviceSynchronize(); };
        float t = (device=="cpu") ? time_cpu_ms(fn_prefill,1,3) : time_cuda_ms(fn_prefill,1,5);
        print_speed("RoPE", device, "prefill", N, t);

        Tensor v({NUM_HEADS, HEAD_DIM}, device);
        fill_random(v, rng, 0.1f);
        auto fn_decode = [&](){ for(int i=0;i<128;i++) RoPE::forward_vec(v, i, ROPE_BASE); if(device=="cuda") cudaDeviceSynchronize(); };
        float t2 = (device=="cpu") ? time_cpu_ms(fn_decode,1,3) : time_cuda_ms(fn_decode,1,3);
        print_speed("RoPE", device, "decode(128)", N, t2);
    }
}

void test_rope(bool do_correct, bool do_speed) {
    if (do_correct) {
        test_rope_prefill_device("cpu");
        test_rope_decode_device("cpu");
        if (cuda_available()) {
            test_rope_prefill_device("cuda");
            test_rope_decode_device("cuda");
        } else test_skip("RoPE CUDA tests (no GPU)");
    }
    if (do_speed) {
        speed_rope_device("cpu");
        if (cuda_available()) speed_rope_device("cuda");
    }
}
