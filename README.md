# 0.7B 对话模型推理引擎

### 概述

该项目是一个为自研的0.7B对话模型（[fancyovo/XingLing](https://github.com/fancyovo/XingLing)）手工打造的 **C++推理引擎**。

该项目的目的在于**初步学习CUDA编程并熟悉经典Transformer中的各种细节**

项目的编写和调试均在 RTX 3060 上完成

### 最终效果

- 手工打造出 **C++/CUDA 推理引擎**。**核心算子全部手写**（仅依赖标准库和 CUDA）。且支持 CPU （包含AVX2指令集和多线程加速）
- Linear层实现了寄存器级的分块GEMM，多头注意力层的实现类似FlashAttention1
- 实现了 KV cache。prefill 默认重置KV cache，decode 默认增量更新
- **RTX 3060** 上速度达到  **prefill 3000 tokens/s**、**decode 80 tokens/s** 的性能
- 项目未实现 Tokenizer，因此利用 pybind11 将 Transformer 编译为 python 库。

### 快速开始

**1. 下载该仓库**

```bash
git clone https://github.com/fancyovo/XingLing-cpp.git
cd XingLing-cpp
```

**2. 安装必要依赖**

```bash
pip install torch transformers
```

**3. 下载并处理训练好的模型权重**

```bash
# 默认下载路径为 data/ , 可通过修改 MODEL_PATH 参数和  EXPORT_PATH 参数来修改路径
python script/export_weights.py
```

**4. 完成一次推理**

```python
# 必须先import torch，用于固定好一些cuda依赖。否则llm_cpp_engine自带的cuda依赖会有冲突
import torch    
import llm_cpp_engine
from transformers import AutoTokenizer

TOKENIZER_PATH = "Qwen3-8B-Tokenizer"
MODEL_PATH = "data/model"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
engine = llm_cpp_engine.InferenceEngine('cuda')
engine.load(MODEL_PATH)

print("model loaded successfully")

text = "爱因斯坦是谁"
text_ids = tokenizer.encode(text)
input_ids = (   [engine.im_start_id, engine.role_id_user, engine.nl_id] + 
                text_ids + 
                [engine.im_end_id, engine.nl_id] +
                [engine.im_start_id, engine.role_id_assistant, engine.nl_id])

token = engine.next_token(input_ids)
output_ids = input_ids + [token]


for i in range(500):
    token = engine.next_token(token)
    if token == engine.im_end_id:
        break
    output_ids.append(token)

output_text = tokenizer.decode(output_ids)
print(output_text)
```

示例输出：

```
model loaded successfully
<|im_start|>user
爱因斯坦是谁<|im_end|>
<|im_start|>assistant
爱因斯坦是20世纪最著名的物理学家之一，他超酷的！他提出了相对论，还搞了实验卫星啥的，但最牛的是提出了狭义相对论。嘿嘿，我虽然看不懂他的理论，但能想象他坐在火箭里飞天的感觉～
```

### 接口清单

仅实现了一个类 `llm_cpp_engine.InferenceEngine`，初始化参数为 `'cpu'` 或 `'cuda'` 。实例化的方法形如

```python
import torch    
import llm_cpp_engine
engine = llm_cpp_engine.InferenceEngine('cuda')
```

实现的接口如下：

- `engine.load(path : string)` ： 加载经过 export 后的模型参数

- `engine.next_token(tokens : list) -> int` ：输入一个token list，默认重置 KV cache，返回预测的下一个token

- `engine.next_token(token : int) -> int` ：输入token，默认增加KV cache，返回预测的下一个token

- `next_token` 支持设定 `temperature`、`top_p`、`repetition_penalty` 三个参数，默认值为

  ```python
  engine.next_token(
  	token : int, # 或 input_ids : list
      temperature = 0.7,
      top_p = 0.9,
      repetition_penalty = 1.1
  )
  ```

- 特殊字符：

  - `engine.im_start_id`
  - `engine.im_end_id`
  - `engine.role_id_user`
  - `engine.role_id_assistant`
  - `engine.nl_id`

- `engine.to(device:string)` 、`engine.device()` ：切换推理设备、获取当前设备

### 模型信息

| 参数                     | 数值     |
| :----------------------- | :------- |
| Layers (层数)            | 16       |
| Hidden Size (隐藏层维度) | 1536     |
| FFN Dimension            | 4096     |
| Heads (注意力头数)       | 24       |
| Max Sequence Length      | 768      |
| Total Parameters         | ~0.68B   |
| Tokenizer                | Qwen3-8B |

详细内容见[fancyovo/XingLing](https://github.com/fancyovo/XingLing)
