# XingLing-cpp (星灵 C++ 推理引擎)

<div align="center">

![C++](https://img.shields.io/badge/C++-17-blue)
![Build](https://img.shields.io/badge/Build-CMake-green)
![Optimization](https://img.shields.io/badge/Arch-AVX2%20%7C%20OpenMP-orange)
![Python Binding](https://img.shields.io/badge/Bindings-pybind11-yellow)

**一个完全不依赖 PyTorch/ONNX/TensorRT，从第一性原理出发手写的 LLM 推理框架。**

[特性](#-特性) | [构建与运行](#-构建与运行) | [架构解析](#-系统架构) | [原模型仓库](https://github.com/fancyovo/XingLing)

</div>

## 📖 简介

**XingLing-cpp** 是 XingLing-Chat-0.68B 模型的纯 C++ 高性能推理后端。

作为初涉工程代码的开发者，我不满足于仅仅调用现成的推理框架（如 llama.cpp 或 vLLM）。为了真正理解大模型底层的计算细节与内存管理，我选择**从零开始（From Scratch）**构建整个技术栈：

*   **没有 `torch.Tensor`**：我手动实现了一个支持广播机制、步长（stride）计算和引用计数的 N 维张量库。
*   **没有 `torch.matmul`**：我基于 AVX2 指令集和 OpenMP 并行化，手写了矩阵乘法、RMSNorm、RoPE 和 Softmax 算子。
*   **没有 `HuggingFace`**：我手动实现了 Transformer 的完整计算图、KV Cache 管理以及 Top-p 采样策略。

本项目旨在展示如何通过算法思维优化系统性能，在 CPU 端实现流畅的 LLM 推理。

## ✨ 特性

*   **纯粹的 C++ 实现**: 核心逻辑不依赖任何第三方深度学习库。
*   **极致的底层优化**:
    *   🔥 **SIMD 加速**: 关键算子（Gemm, RMSNorm, RoPE, DotProduct）均使用 **AVX2 (`_mm256`)** 指令集重写。
    *   ⚡ **并行计算**: 利用 **OpenMP** 实现算子级的多线程并行。
    *   💾 **内存管理**: 手写的 Tensor 系统，支持 Zero-copy 的 reshape/transpose 操作；实现了 KV Cache 的预分配与复用。
*   **完整的 LLM 功能**:
    *   支持 ChatML 格式对话 (`<|im_start|>`, `<|im_end|>`)。
    *   支持 Top-p (Nucleus) 采样、Temperature 调节、Repetition Penalty。
    *   实现了 Prefill（预填充）与 Decode（解码）分离的 Attention 逻辑。
*   **Python 绑定**: 通过 `pybind11` 提供 Python 接口，可直接与 `transformers` 的 Tokenizer 配合使用。

## 📂 项目结构

```
XingLing-cpp
├── include/              # 头文件 (Tensor接口, 算子定义, 引擎API)
├── src/
│   ├── tensor.cpp        # 核心张量库实现 (Strides, Broadcasting, Memory)
│   ├── ops.cpp           # AVX2/OpenMP 加速的高性能算子实现
│   ├── inference.cpp     # Transformer 架构、KV Cache 与 采样策略
│   └── bindings.cpp      # pybind11 Python 接口封装
├── scripts/
│   └── export_weights.py # 权重转换脚本 (pth -> bin)
├── CMakeLists.txt        # 构建脚本 (自动检测 AVX2/OpenMP)
└── main.py               # 推理演示入口
```

## 🚀 构建与运行

### 1. 环境准备

确保你的系统安装了 C++ 编译器（支持 C++17）、CMake 和 Python。

```bash
# 安装 pybind11 (用于生成 Python 扩展)
pip install pybind11 torch transformers
```

### 2. 克隆仓库

```bash
git clone https://github.com/fancyovo/XingLing-cpp.git
cd XingLing-cpp
```

### 3. 权重转换

由于本引擎不使用 PyTorch 加载权重，你需要先将 PyTorch 的 `.pth` 权重导出为本引擎可读取的纯二进制 `.bin` 格式。

```bash
# 该脚本会自动下载 fancyovo/XingLing-Chat-0.68B-SFT 模型并转换
python scripts/export_weights.py
```
*转换后的权重将保存在 `data/model` 目录下。*

### 4. 编译 C++ 引擎

使用 CMake 进行构建。构建脚本会自动检测你的环境（Linux GCC 或 Windows MSVC）并开启 AVX2 和 OpenMP 优化。

```bash
mkdir build
cd build
cmake ..
make -j4  # Windows 用户请使用 cmake --build . --config Release
```
编译成功后，会在项目根目录生成 `llm_cpp_engine.cpython-xxx.so` (Linux) 或 `.pyd` (Windows) 文件。

### 5. 运行推理

回到项目根目录，运行 `main.py` 即可体验对话：

```bash
python main.py
```

**运行输出示例：**

```
model loaded successfully
<|im_start|>user
天空是什么颜色<|im_end|>
<|im_start|>assistant
天空是蓝色的，看起来像太阳照在海面上。它看起来像一块大大的黑色布料，反射阳光，看起来超梦幻！
```

## 🧩 系统架构细节

### 1. Tensor System (张量系统)
为了理解 PyTorch 的底层，我手写了 `Tensor` 类。它采用了**数据与视图分离**的设计：
*   支持任意维度的 `reshape` 和 `transpose`，且不发生内存拷贝（通过修改 strides 实现）。
*   实现了 `contiguous()` 方法，在需要连续内存进行 SIMD 优化时自动重整数据。
*   实现了完整的 **Broadcasting (广播)** 机制，支持不同形状张量的四则运算。

### 2. Operator Optimization (算子优化)
推理的性能瓶颈在于矩阵乘法和访存。
*   **矩阵乘法**: 针对 LLM 解码阶段 `(Batch, 1, In_Dim) * (In_Dim, Out_Dim)` 的特征，实现了特定的向量化 GEMV。
*   **RoPE**: 预计算旋转矩阵，利用复数乘法的几何性质，配合 AVX2 实现快速位置编码注入。
*   **Softmax**: 实现了 Online Softmax 算法，结合 OpenMP 并行，数值稳定性更强且访存更少。

### 3. KV Cache 管理
*   预分配了 `(Max_Seq_Len, Num_Heads, Head_Dim)` 的显存空间。
*   使用 `std::copy` 进行 Cache 更新，避免了 Python 中常见的 `torch.cat` 带来的内存碎片和拷贝开销。
*   将 Attention 拆分为 `prefill` (并行计算) 和 `forward` (单步递推) 两个函数，最大化预处理阶段的并行度。

## 🤝 致谢与引用

*   原模型训练仓库：XingLing (星灵) [https://github.com/fancyovo/XingLing](https://github.com/fancyovo/XingLing)
*   Tokenizer 支持：Qwen [https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen)

如果你对这个项目感兴趣，或者它对你的学习有帮助，请给个 Star ⭐️！

```bibtex
@misc{XingLingCpp2025,
  author = {fancyovo},
  title = {XingLing-cpp: A High-Performance Pure C++ Inference Engine for LLMs},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fancyovo/XingLing-cpp}}
}
```

## ⚖️ License

Apache 2.0 License