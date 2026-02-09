#include "../include/tensor.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

// --- 简单的工程化测试辅助函数 ---

void fail(const std::string& msg) {
    std::cerr << "\033[31m[FAILED] " << msg << "\033[0m" << std::endl;
    exit(1);
}

void expect_float(float actual, float expected, const std::string& name) {
    if (std::abs(actual - expected) > 1e-5) {
        fail(name + ": Expected " + std::to_string(expected) + ", but got " + std::to_string(actual));
    }
}

void expect_int(int actual, int expected, const std::string& name) {
    if (actual != expected) {
        fail(name + ": Expected " + std::to_string(expected) + ", but got " + std::to_string(actual));
    }
}

void expect_vector(const std::vector<int>& actual, const std::vector<int>& expected, const std::string& name) {
    if (actual.size() != expected.size()) {
        fail(name + ": Vector size mismatch");
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            fail(name + " at index " + std::to_string(i) + ": Expected " + std::to_string(expected[i]) + ", got " + std::to_string(actual[i]));
        }
    }
}

// --- 具体测试用例 ---

void test_memory_logic() {
    std::cout << "Running: test_memory_logic..." << std::endl;
    Tensor t1({2, 2});
    t1[{0, 0}] = 1.0f; t1[{0, 1}] = 2.0f;
    t1[{1, 0}] = 3.0f; t1[{1, 1}] = 4.0f;

    // 测试默认构造是浅拷贝 (ICPC选手要注意，工程中这叫 Alias)
    Tensor t2 = t1; 
    t2[{0, 0}] = 99.0f;
    expect_float(t1[{0, 0}], 99.0f, "Shallow copy (Alias) test");

    // 测试 clone 是深拷贝
    Tensor t3 = t1.clone();
    t3[{0, 0}] = 1.0f;
    expect_float(t1[{0, 0}], 99.0f, "Deep copy (Clone) test - original should not change");
    expect_float(t3[{0, 0}], 1.0f, "Deep copy (Clone) test - clone should change");
}

void test_strides_and_transpose() {
    std::cout << "Running: test_strides_and_transpose..." << std::endl;
    // 创建一个 [2, 3] Tensor
    // 逻辑形状 [2, 3], 默认步长应该是 [3, 1]
    Tensor t({2, 3});
    expect_vector(t.strides_, {3, 1}, "Initial strides check");

    // 转置为 [3, 2]
    Tensor tt = t.transpose({1, 0});
    expect_vector(tt.shape(), {3, 2}, "Transpose shape check");
    expect_vector(tt.strides_, {1, 3}, "Transpose strides check"); // 步长应该互换
    
    // 转置后应该是非连续的
    if (tt.is_contiguous()) fail("Transposed tensor should not be contiguous");

    // 转连续化
    Tensor tc = tt.contiguous();
    expect_vector(tc.shape(), {3, 2}, "Contiguous shape check");
    expect_vector(tc.strides_, {2, 1}, "Contiguous strides check");
}

void test_reshape() {
    std::cout << "Running: test_reshape..." << std::endl;
    Tensor t({2, 3, 4}); // size = 24
    Tensor r = t.reshape({4, 6});
    expect_int(r.size(), 24, "Reshape size consistency");
    expect_vector(r.shape(), {4, 6}, "Reshape shape check");
    expect_vector(r.strides_, {6, 1}, "Reshape strides check");
}

void test_broadcasting_complex() {
    std::cout << "Running: test_broadcasting_complex..." << std::endl;
    /*
      Case: [2, 1, 3] + [1, 4, 3] -> Result [2, 4, 3]
      这是广播中最复杂的情况之一
    */
    Tensor a({2, 1, 3});
    Tensor b({1, 4, 3});

    // 填充 a: a[i, 0, k] = 1.0
    for(int i=0; i<2; i++) for(int k=0; k<3; k++) a[{i, 0, k}] = 10.0f * (i+1);
    
    // 填充 b: b[0, j, k] = 1.0
    for(int j=0; j<4; j++) for(int k=0; k<3; k++) b[{0, j, k}] = (float)k;

    Tensor c = a + b;

    expect_vector(c.shape(), {2, 4, 3}, "Broadcast result shape");
    
    // 验证特定点: c[1, 2, 2] = a[1, 0, 2] + b[0, 2, 2] = 20.0 + 2.0 = 22.0
    expect_float(c[{1, 2, 2}], 22.0f, "Complex broadcast value check at [1, 2, 2]");
    expect_float(c[{0, 3, 0}], 10.0f, "Complex broadcast value check at [0, 3, 0]");
}

void test_scalar_ops() {
    std::cout << "Running: test_scalar_ops..." << std::endl;
    Tensor t({2, 2});
    for(int i=0; i<4; i++) t.data_.get()[i] = 2.0f;
    
    Tensor res = (t * 3.0f) + 4.0f; // 2 * 3 + 4 = 10
    for(int i=0; i<4; i++) {
        expect_float(res.data_.get()[i], 10.0f, "Scalar op check");
    }
}

int main() {
    test_memory_logic();
    test_strides_and_transpose();
    test_reshape();
    test_broadcasting_complex();
    test_scalar_ops();

    std::cout << "\n\033[32m[PASSED] All comprehensive tests passed!\033[0m" << std::endl;
    return 0;
}
