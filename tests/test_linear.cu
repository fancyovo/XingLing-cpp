#include "tests.h"
#include <cmath>

static void naive_linear(
    const float* input, const float* weight, const float* bias,
    float* output, int N, int M, int K
){
    for (int i=0;i<N;i++){
        for (int j=0;j<M;j++){
            float sum = bias[j];
            for (int k=0;k<K;k++){
                sum += input[i*K + k] * weight[j*K + k];
            }
            output[i*M + j] = sum;
        }
    }
}

static void naive_linear_vec(
    const float* input, const float* weight, const float* bias,
    float* output, int M, int K
){
    for (int j=0;j<M;j++){
        float sum = bias[j];
        for (int k=0;k<K;k++){
            sum += input[k] * weight[j*K + k];
        }
        output[j] = sum;
    }
}

static void test_linear_prefill_device(const std::string& device) {
    int N=37, K=LATENT_DIM, M=LATENT_DIM;
    std::mt19937 rng(123);

    Linear lin(K, M, device);
    init_linear_weights(lin, rng);

    Tensor input({N,K}, device);
    Tensor output({N,M}, device);
    fill_random(input, rng, 0.1f);

    lin.forward(input, output);
    if (device=="cuda") CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> ref(N*M);
    std::vector<float> input_h = copy_tensor_to_host(input);
    naive_linear(input_h.data(), lin.weight_.h_ptr, lin.bias_.h_ptr,
                 ref.data(), N, M, K);

    std::vector<float> out_h = copy_tensor_to_host(output);
    expect_allclose(out_h, ref, 1e-3f, 1e-3f,
                    "Linear prefill " + device);
}

static void test_linear_decode_device(const std::string& device) {
    int K=LATENT_DIM, M=LATENT_DIM;
    std::mt19937 rng(456);

    Linear lin(K, M, device);
    init_linear_weights(lin, rng);

    Tensor input({K}, device);
    Tensor output({M}, device);
    fill_random(input, rng, 0.1f);

    lin.forward_vec(input, output);
    if (device=="cuda") CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> ref(M);
    std::vector<float> input_h = copy_tensor_to_host(input);
    naive_linear_vec(input_h.data(), lin.weight_.h_ptr, lin.bias_.h_ptr,
                     ref.data(), M, K);

    std::vector<float> out_h = copy_tensor_to_host(output);
    expect_allclose(out_h, ref, 1e-3f, 1e-3f,
                    "Linear decode " + device);
}

static void speed_linear_device(const std::string& device) {
    std::vector<int> lens = {64,128,256,512};
    int K=LATENT_DIM, M=LATENT_DIM;
    std::mt19937 rng(7);

    Linear lin(K, M, device);
    init_linear_weights(lin, rng);

    for (int N: lens) {
        Tensor input({N,K}, device);
        Tensor output({N,M}, device);
        fill_random(input, rng, 0.1f);

        auto fn_prefill = [&](){ lin.forward(input, output); if(device=="cuda") cudaDeviceSynchronize(); };
        float t = (device=="cpu") ? time_cpu_ms(fn_prefill,1,3) : time_cuda_ms(fn_prefill,1,5);
        print_speed("Linear", device, "prefill", N, t);

        // decode: 128 steps
        Tensor vin({K}, device);
        Tensor vout({M}, device);
        fill_random(vin, rng, 0.1f);
        auto fn_decode = [&](){
            for (int i=0;i<128;i++) lin.forward_vec(vin, vout);
            if(device=="cuda") cudaDeviceSynchronize();
        };
        float t2 = (device=="cpu") ? time_cpu_ms(fn_decode,1,3) : time_cuda_ms(fn_decode,1,3);
        print_speed("Linear", device, "decode(128)", N, t2);
    }
}

void test_linear(bool do_correct, bool do_speed) {
    if (do_correct) {
        test_linear_prefill_device("cpu");
        test_linear_decode_device("cpu");
        if (cuda_available()) {
            test_linear_prefill_device("cuda");
            test_linear_decode_device("cuda");
        } else {
            test_skip("Linear CUDA tests (no GPU)");
        }
    }
    if (do_speed) {
        speed_linear_device("cpu");
        if (cuda_available()) speed_linear_device("cuda");
    }
}
