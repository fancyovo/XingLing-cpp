#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
// 检查CUDA错误的宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(1); \
        } \
    } while(0)
// 检查 Kernel 启动是否成功的宏（轻量级）
#define CHECK_KERNEL_LAUNCH() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "Kernel Launch Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)
// 检查 Kernel 是否运行成功的宏（重量级，会同步，建议只在 Debug 或特定地方用）
#define CHECK_KERNEL_SYNC() \
    do { \
        cudaDeviceSynchronize(); \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "Kernel Sync Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)
inline void __print_vec__(const std::vector<int>& vec, std::string name) {
    if (vec.empty()) {
        std::cout << name << ": ()" << std::endl;
        return;
    }
    std::cout << name << ": "<<"("<<vec[0];
    for (int i = 1; i < vec.size(); i++) {
        std::cout << ", " << vec[i];
    }
    std::cout << ")" << std::endl;
}

inline bool checkGpuSupport(bool show = false) {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        if (show) {
            std::cerr << "CUDA 探测失败: " << cudaGetErrorString(error) << std::endl;
            std::cerr << "可能原因：未安装 NVIDIA 驱动，或驱动版本过旧。" << std::endl;
        }
        return false;
    }
    if (deviceCount == 0) {
        if (show) std::cerr << "未找到支持 CUDA 的 GPU 设备。" << std::endl;
        return false;
    }
    if (show) std::cout << "检测到 " << deviceCount << " 个 GPU 设备。" << std::endl;
    return true;
}