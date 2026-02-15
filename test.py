import torch
import llm_cpp_engine
from transformers import AutoTokenizer
import time

# ================= 配置区域 =================
TOKENIZER_PATH = "Qwen3-8B-Tokenizer"
MODEL_PATH = "data/model"
DEVICE = "cuda"  # 使用 GPU 加速测试

# 测试用例：(问题, 最大生成长度)
# 注意：最大长度设置大一些，防止回答被截断
TEST_CASES = [
    ("你好，请介绍一下你自己。", 100),
    ("1+1等于几？", 50),
    ("中国的首都是哪里？", 50),
    ("天空是什么颜色的？", 50),
    ("写一首关于春天的诗。", 100)
]

# 温度对比设置
TEMPERATURES = [0.2, 0.7]
# ============================================

def format_chat_prompt(engine, tokenizer, user_query):
    """
    手动构建 ChatML 格式的 Prompt ID 序列
    假设格式为: <im_start>user\n{query}<im_end>\n<im_start>assistant\n
    """
    # 1. 编码用户问题
    # 注意：有些 tokenizer 会自动加 special token，有些不会。
    # 为了配合你的 C++ 引擎，我们尽量手动控制。
    query_ids = tokenizer.encode(user_query, add_special_tokens=False)
    
    # 2. 拼接前缀
    # 根据你 main.py 里的逻辑：
    # prefix = [im_start, role_user, nl] + text + [im_end, nl, im_start, role_assistant, nl]
    
    input_ids = (
        [engine.im_start_id, engine.role_id_user, engine.nl_id] + 
        query_ids + 
        [engine.im_end_id, engine.nl_id] +
        [engine.im_start_id, engine.role_id_assistant, engine.nl_id]
    )
    return input_ids

def run_quality_test():
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    print(f"Loading model on {DEVICE}...")
    engine = llm_cpp_engine.InferenceEngine(DEVICE)
    engine.load(MODEL_PATH)
    print("Model loaded.\n")

    print("="*30 + " 开始质量测试 " + "="*30)

    for i, (question, max_len) in enumerate(TEST_CASES):
        print(f"\n[Question {i+1}]: {question}")
        
        # 准备 Prompt 的 Input IDs
        prompt_ids = format_chat_prompt(engine, tokenizer, question)
        
        for temp in TEMPERATURES:
            print(f"  --- [Temp: {temp}] ---")
            
            # 1. 重置 KV Cache 并进行 Prefill (传入 list)
            # 这一步会消耗一些时间
            start_token = engine.next_token(prompt_ids, temperature=temp)
            
            # 2. 开始生成
            output_ids = [start_token]
            
            # 简单的计时
            t_start = time.time()
            
            for step in range(max_len):
                # 传入 int，进行 Decode (追加 KV Cache)
                next_id = engine.next_token(int(output_ids[-1]), temperature=temp)
                
                # 遇到结束符则停止
                if next_id == engine.im_end_id:
                    break
                    
                output_ids.append(next_id)

            t_end = time.time()
            
            # 3. 解码并打印结果
            # 解码时跳过 prompt 部分，只看生成的部分
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # 清理一下可能的乱码尾部
            text = text.strip()
            
            # 计算速度
            gen_len = len(output_ids)
            speed = gen_len / (t_end - t_start) if (t_end - t_start) > 0 else 0
            
            print(f"  [Answer]: {text}")
            print(f"  [Stats]: Generated {gen_len} tokens. Speed: {speed:.2f} tokens/s")

if __name__ == "__main__":
    run_quality_test()
