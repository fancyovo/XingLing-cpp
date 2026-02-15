import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import llm_cpp_engine  # 你的C++编译出来的库

# ================= 配置区域 =================
TOKENIZER_PATH = "Qwen3-8B-Tokenizer"  # 请确保路径正确
MODEL_PATH = "data/model"              # 请确保路径正确

# 测试配置
WARMUP_STEPS = 5          # 预热步数（让GPU进入高频状态）
MAX_SEQ_LEN = 750         # 测试的最大序列长度
STEP = 50                 # 采样的序列长度步长
DECODE_GEN_LEN = 8        # 每次Decode测试生成多少个token来取平均值
# ===========================================

def get_dummy_input(tokenizer, length):
    """
    生成指定长度的dummy input
    实际场景中你可以用真实文本，但为了控制变量（严格测量长度影响），用随机ID更科学
    """
    # 为了避免SPECIAL TOKEN打断逻辑，我们只取常见字或随机ID
    # 这里简单起见，假设词表大小并随机生成，或者重复某个字符串
    # 保持和你的代码逻辑一致：直接生成 list[int]
    return np.random.randint(100, 10000, size=length).tolist()

def benchmark_prefill(engine, input_ids):
    """
    测量 Prefill 速度 (Time To First Token)
    即：处理整个 input_ids 并生成第一个 token 的时间
    """
    start = time.perf_counter()
    # next_token(list) 触发 prefill
    engine.next_token(input_ids) 
    end = time.perf_counter()
    return (end - start) * 1000  # 返回毫秒

def benchmark_decode(engine, input_ids, gen_len):
    """
    测量 Decode 速度
    1. 先 Prefill (构建 KV Cache)
    2. 循环生成 gen_len 个 token，计算平均耗时
    """
    # 1. Prefill (不计入时间)
    first_token = engine.next_token(input_ids)
    current_token = first_token
    
    # 2. Warm up within decode (可选，这里为了准确性只测主体)
    # 3. Timed Decode Loop
    times = []
    for _ in range(gen_len):
        start = time.perf_counter()
        # next_token(int) 触发 decode (追加到 kv_cache)
        current_token = engine.next_token(current_token) 
        end = time.perf_counter()
        times.append((end - start) * 1000)
        
    return np.mean(times) # 返回平均单步耗时

def run_test_for_device(device_name):
    print(f"\n{'='*20} Testing on {device_name.upper()} {'='*20}")
    
    # 1. 初始化引擎
    engine = llm_cpp_engine.InferenceEngine(device_name)
    engine.load(MODEL_PATH)
    
    # 2. Warmup (非常关键!)
    # GPU 需要预热：CUDA Context初始化、显存分配、Kernel编译(JIT)、频率提升
    print(f"Warming up {device_name}...")
    warmup_ids = [100, 200, 300] 
    try:
        # 随便跑几次
        for _ in range(WARMUP_STEPS):
            engine.next_token(warmup_ids)
            engine.next_token(100) # decode step
    except Exception as e:
        print(f"Warmup failed: {e}")
        return None

    # 3. 准备数据收集
    seq_lengths = list(range(STEP, MAX_SEQ_LEN + 1, STEP))
    prefill_times = []
    decode_times = []
    
    # 4. 循环测试
    print(f"Starting benchmark (Max Len: {MAX_SEQ_LEN})...")
    for length in seq_lengths:
        # 生成输入
        input_ids = get_dummy_input(None, length)
        
        # A. 测 Prefill
        t_prefill = benchmark_prefill(engine, input_ids)
        prefill_times.append(t_prefill)
        
        # B. 测 Decode
        # 注意：执行测 Decode 时，seq_len 实际上已经增加了1 (因为刚才 Prefill 了)
        # 但对于长序列趋势图，这 1 的差异可以忽略
        t_decode = benchmark_decode(engine, input_ids, DECODE_GEN_LEN)
        decode_times.append(t_decode)
        
        print(f"Len: {length:3d} | Prefill: {t_prefill:6.2f} ms | Decode: {t_decode:6.2f} ms/token")

    return seq_lengths, prefill_times, decode_times

def plot_results(results_cpu, results_gpu):
    if not results_cpu and not results_gpu:
        print("No data to plot.")
        return

    # 创建 2x2 的子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Inference Engine Performance Analysis', fontsize=16)
    
    # ---------- 图 1: CPU Prefill ----------
    if results_cpu:
        x, y, _ = results_cpu
        axs[0, 0].plot(x, y, marker='o', color='blue', label='CPU Prefill')
        axs[0, 0].set_title('CPU Prefill Time vs Seq Len')
        axs[0, 0].set_xlabel('Sequence Length')
        axs[0, 0].set_ylabel('Time (ms)')
        axs[0, 0].grid(True)
        # 设置 Y 轴从 0 开始，方便看量级
        axs[0, 0].set_ylim(bottom=0)

    # ---------- 图 2: GPU Prefill ----------
    if results_gpu:
        x, y, _ = results_gpu
        axs[0, 1].plot(x, y, marker='o', color='green', label='GPU Prefill')
        axs[0, 1].set_title('GPU Prefill Time vs Seq Len')
        axs[0, 1].set_xlabel('Sequence Length')
        axs[0, 1].set_ylabel('Time (ms)')
        axs[0, 1].grid(True)
        axs[0, 1].set_ylim(bottom=0)

    # ---------- 图 3: CPU Decode ----------
    if results_cpu:
        x, _, y = results_cpu
        axs[1, 0].plot(x, y, marker='s', color='red', label='CPU Decode')
        axs[1, 0].set_title('CPU Decode Time vs Seq Len')
        axs[1, 0].set_xlabel('Sequence Length (Context)')
        axs[1, 0].set_ylabel('Time per Token (ms)')
        axs[1, 0].grid(True)
        axs[1, 0].set_ylim(bottom=0)

    # ---------- 图 4: GPU Decode ----------
    if results_gpu:
        x, _, y = results_gpu
        axs[1, 1].plot(x, y, marker='s', color='orange', label='GPU Decode')
        axs[1, 1].set_title('GPU Decode Time vs Seq Len')
        axs[1, 1].set_xlabel('Sequence Length (Context)')
        axs[1, 1].set_ylabel('Time per Token (ms)')
        axs[1, 1].grid(True)
        axs[1, 1].set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("performance_benchmark.png")
    print("\nPlot saved to performance_benchmark.png")
    plt.show()

if __name__ == "__main__":
    # 检测是否有 GPU (模拟你的逻辑)
    # 由于你的库是自行编译的，我们通过 try-catch 来跑
    res_cpu = None
    res_gpu = None

    # 1. CPU 测试
    try:
        res_cpu = run_test_for_device("cpu")
    except Exception as e:
        print(f"CPU Test Failed: {e}")

    # 2. GPU 测试
    # 注意：如果你的 C++ 代码里检测不到 CUDA，这里会报错
    try:
        res_gpu = run_test_for_device("cuda")
    except Exception as e:
        print(f"GPU Test Failed: {e}")

    # 3. 绘图
    plot_results(res_cpu, res_gpu)
