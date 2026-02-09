import time
import math
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import llm_cpp_engine

# ================= 配置区域 =================
TOKENIZER_PATH = "Qwen3-8B-Tokenizer"
MODEL_PATH = "data/model"

# 性能优化参数
MAX_SEQ_LEN = 768          # 模型支持的最大长度
PPL_MAX_TOKENS = 4096      # PPL评测最多用多少token（避免太慢）
PPL_STRIDE = 512           # 滑动窗口步长
PPL_MAX_WINDOWS = 8        # 最多算8个窗口，保证速度

# C-Eval 评测参数
# 选几个有代表性的学科：计算机、历史、物理、语文、法律
CEVAL_SUBJECTS = [
    'computer_network', 
    'operating_system', 
    'chinese_history', 
    'high_school_physics',
    'high_school_chinese',
    'tax_law'
]
CEVAL_SAMPLES_PER_SUBJ = 25  # 每个学科测25题（全量太慢，25题已能反映水平）

class ChineseBenchmark:
    def __init__(self):
        print("Loading tokenizer and engine...")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self.engine = llm_cpp_engine.InferenceEngine()
        self.engine.load(MODEL_PATH)
        print("Model loaded successfully.\n")
        
    def evaluate_speed(self, prompt="请介绍一下北京故宫的历史：", gen_len=100):
        """测试首字延迟和生成吞吐"""
        print("=== Speed Benchmark ===")
        input_ids = self.tokenizer.encode(prompt)
        
        # Warmup
        for _ in range(3):
            self.engine.next_token(input_ids)
            
        # Prefill (处理prompt)
        t0 = time.time()
        first_token = self.engine.next_token(input_ids)
        t_prefill = time.time() - t0
        
        # Decode (生成阶段)
        t0 = time.time()
        current_token = first_token
        for _ in range(gen_len):
            current_token = self.engine.next_token(current_token, 0.0)
        t_decode = time.time() - t0
        
        prefill_tps = len(input_ids) / t_prefill
        decode_tps = gen_len / t_decode
        
        print(f"Prompt length: {len(input_ids)} tokens")
        print(f"Prefill speed: {prefill_tps:.2f} tokens/s (latency: {t_prefill*1000:.1f} ms)")
        print(f"Decode speed : {decode_tps:.2f} tokens/s")
        print(f"[Reference] TinyLlama-1.1B on CPU: ~10-20 t/s")
        print()
        return decode_tps

    def evaluate_ppl_ceval(self):
        """
        用 C-Eval 验证集的文本计算中文 PPL。
        将 C-Eval 的问题和选项拼接成长文本，计算困惑度。
        """
        print("=== Chinese PPL Benchmark (C-Eval corpus) ===")
        print("Loading C-Eval data...")
        
        all_texts = []
        for subject in CEVAL_SUBJECTS:
            try:
                ds = load_dataset("ceval/ceval-exam", subject, split="val")
                for item in ds:
                    # 拼接题干和选项
                    text = item['question']
                    for choice in ['A', 'B', 'C', 'D']:
                        text += f" {choice}.{item[choice]}"
                    all_texts.append(text)
            except Exception as e:
                print(f"Warning: failed to load {subject}: {e}")
                continue
        
        if not all_texts:
            print("Error: No C-Eval data loaded. Check internet connection.")
            return 0.0
            
        # 拼接并截断
        full_text = "\n".join(all_texts)
        encodings = self.tokenizer(
            full_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=PPL_MAX_TOKENS
        )
        all_ids = encodings.input_ids[0].tolist()
        seq_len = len(all_ids)
        
        nlls = []
        # 滑动窗口计算，但限制窗口数量以保证速度
        step = 0
        for begin_loc in range(0, seq_len, PPL_STRIDE):
            if step >= PPL_MAX_WINDOWS:
                break
                
            end_loc = min(begin_loc + MAX_SEQ_LEN, seq_len)
            chunk = all_ids[begin_loc:end_loc]
            
            if len(chunk) < 2:
                break
                
            # input: 除最后一个token外所有
            # target: 除第一个token外所有（即整体右移一位）
            input_chunk = chunk[:-1]
            target_chunk = chunk[1:]
            
            # 调用C++接口
            avg_nll = self.engine.nll(input_chunk, target_chunk)
            nlls.append(avg_nll)
            step += 1
            
        if not nlls:
            return 0.0
            
        mean_nll = np.mean(nlls)
        ppl = math.exp(mean_nll)
        
        print(f"Evaluated on {step} windows, ~{step * len(chunk)} tokens")
        print(f"Average NLL: {mean_nll:.4f}")
        print(f"PPL: {ppl:.2f}")
        print(f"[Reference] SkyPile-trained 0.7B model expected: 15-30")
        print(f"[Reference] Random/untuned model: >100")
        print()
        return ppl

    def evaluate_ceval_accuracy(self):
        """
        C-Eval 多项选择题评测 (0-shot)。
        使用 Log-Likelihood 选择答案，而非生成文本。
        这对小模型更公平，避免生成格式混乱导致的误判。
        """
        print("=== C-Eval Accuracy Benchmark (0-shot) ===")
        print(f"Testing on {len(CEVAL_SUBJECTS)} subjects, {CEVAL_SAMPLES_PER_SUBJ} samples each...")
        print("Note: 0.7B model may be close to random (25%), which is normal for small models.\n")
        
        all_results = []
        
        for subject in CEVAL_SUBJECTS:
            try:
                ds = load_dataset("ceval/ceval-exam", subject, split="val")
            except Exception as e:
                print(f"Skipping {subject}: {e}")
                continue
                
            samples = list(ds)[:CEVAL_SAMPLES_PER_SUBJ]
            correct = 0
            total = 0
            
            for item in tqdm(samples, desc=f"Subject: {subject}", leave=False):
                question = item['question']
                answer = item['answer']  # 'A', 'B', 'C', or 'D'
                
                # 构造prompt模板
                prompt = f"问题：{question}\n"
                for choice in ['A', 'B', 'C', 'D']:
                    prompt += f"{choice}. {item[choice]}\n"
                prompt += "答案："
                
                prompt_ids = self.tokenizer.encode(prompt)
                
                # 计算每个选项的 NLL（负对数似然）
                # NLL越低，说明模型认为该选项概率越高
                scores = {}
                for choice in ['A', 'B', 'C', 'D']:
                    # 将选项字母加入序列
                    choice_ids = self.tokenizer.encode(choice, add_special_tokens=False)
                    full_ids = prompt_ids + choice_ids
                    
                    if len(full_ids) > MAX_SEQ_LEN:
                        full_ids = full_ids[-MAX_SEQ_LEN:]
                        
                    if len(full_ids) < 2:
                        continue
                        
                    # 计算最后一个token（即选项字母）的loss
                    # 输入是除最后一个外的所有token，目标是除第一个外的所有token
                    input_ids = full_ids[:-1]
                    target_ids = full_ids[1:]
                    
                    nll = self.engine.nll(input_ids, target_ids)
                    scores[choice] = nll
                
                if not scores:
                    continue
                    
                # 选择NLL最小的作为预测
                pred = min(scores, key=scores.get)
                if pred == answer:
                    correct += 1
                total += 1
            
            if total > 0:
                acc = correct / total
                all_results.append(acc)
                print(f"{subject:25s}: {acc*100:5.1f}% ({correct:2d}/{total:2d})")
            else:
                print(f"{subject:25s}: No valid samples")
        
        if all_results:
            avg_acc = np.mean(all_results) * 100
            print(f"\nAverage Accuracy: {avg_acc:.1f}%")
            print(f"[Reference] Random guess: 25.0%")
            print(f"[Reference] ChatGLM3-6B: ~60%, Qwen2-0.5B: ~40-45%")
            print(f"[Reference] Your 0.7B model: expected 25-40% (depending on training data diversity)")
        else:
            print("No results generated.")
            
        return all_results

if __name__ == "__main__":
    runner = ChineseBenchmark()
    
    # 1. 速度测试
   # runner.evaluate_speed()
    
    # 2. 中文PPL（基于C-Eval语料）
    runner.evaluate_ppl_ceval()
    
    # 3. C-Eval选择题准确率
    runner.evaluate_ceval_accuracy()
