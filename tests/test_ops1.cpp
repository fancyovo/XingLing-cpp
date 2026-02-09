#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <string>
#include <functional>
#include "../include/tensor.h"
#include "../include/ops.h"

using namespace std;

// 辅助函数：判断浮点数是否足够接近
bool is_close(float a, float b, float epsilon = 1e-4) {
    return std::abs(a - b) < epsilon;
}

// 辅助函数：填充随机数
void fill_random(Tensor& t) {
    for (int i = 0; i < t.size(); i++) {
        t.data_[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
}

// 统一的性能测试框架
void run_benchmark(string name, function<void()> op, long long total_ops, int iters = 10) {
    // 预热
    op();
    
    auto start = chrono::high_resolution_clock::now();
    for(int i=0; i<iters; i++) {
        op();
    }
    auto end = chrono::high_resolution_clock::now();
    
    chrono::duration<double> diff = end - start;
    double avg_time = diff.count() / iters;
    double gflops = (total_ops) / (avg_time * 1e9);
    
    cout << left << setw(20) << name 
         << " | Time: " << fixed << setw(10) << setprecision(6) << avg_time << " s"
         << " | Performance: " << setw(10) << setprecision(2) << gflops << " GFLOPS/GBs" << endl;
}

void test_correctness() {
    cout << "=== Correctness Tests ===" << endl;
    Functions ops;

    // 1. Test Relu
    Tensor t1({4});
    t1.data_[0] = -1.0; t1.data_[1] = 0.0; t1.data_[2] = 1.0; t1.data_[3] = 2.0;
    Tensor res_relu = ops.relu(t1);
    bool relu_ok = (res_relu.data_[0] == 0 && res_relu.data_[3] == 2.0);
    cout << "ReLU:    " << (relu_ok ? "[PASS]" : "[FAIL]") << endl;

    // 2. Test Dot Product
    Tensor da({8}), db({8});
    for(int i=0; i<8; i++) { da.data_[i] = 1.0f; db.data_[i] = 2.0f; }
    Tensor res_dot = ops.dot(da, db); 
    // res_dot 应为 1*2*8 = 16.0
    cout << "Dot:     " << (is_close(res_dot.data_[0], 16.0f) ? "[PASS]" : "[FAIL]") << " Value: " << res_dot.data_[0] << endl;

    // 3. Test Softmax
    Tensor s1({4});
    s1.data_[0] = 1.0; s1.data_[1] = 1.0; s1.data_[2] = 1.0; s1.data_[3] = 1.0;
    Tensor res_soft = ops.softmax(s1);
    // 应该全是 0.25
    cout << "Softmax: " << (is_close(res_soft.data_[0], 0.25f) ? "[PASS]" : "[FAIL]") << endl;
}

void test_performance() {
    cout << "\n=== Performance Benchmarks (N=1024) ===" << endl;
    Functions ops;
    int N = 1024;

    // MatMul Benchmark (Compute Bound)
    // 根据你的定义：A(N, N), B(N, N) -> res(N, N)
    // 总计算量约为 2 * N^3
    Tensor A({N, N}), B({N, N});
    fill_random(A); fill_random(B);
    run_benchmark("MatMul 1024x1024", [&](){ ops.matmul(A, B); }, 2LL * N * N * N);

    // Unary Ops Benchmark (Memory Bound)
    // 这些算子主要测试内存带宽和向量化数学函数
    Tensor X({N * N});
    fill_random(X);
    run_benchmark("Exp (N^2)", [&](){ ops.Exp(X); }, N * N); // 这里 GFLOPS 仅供参考
    run_benchmark("Sigmoid (N^2)", [&](){ ops.sigmoid(X); }, N * N);

    // Dot Product Benchmark
    // 8192个长度为8192的向量做点积
    int D = 8192;
    Tensor D1({D, D}), D2({D, D});
    fill_random(D1); fill_random(D2);
    run_benchmark("Dot 8192x8192", [&](){ ops.dot(D1, D2); }, 2LL * D * D);
}

int main() {
    srand(42);
    test_correctness();
    test_performance();
    return 0;
}
