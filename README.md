# XingLing-Inference (C++/CUDA)

<div align="center">
![C++](https://img.shields.io/badge/C++-17-blue?logo=cplusplus)
![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green?logo=nvidia)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-Apache%202.0-orange)

**A High-Performance, Handcrafted C++/CUDA Inference Engine for the XingLing-0.68B LLM**

[Features](#-key-features) | [Performance](#-performance) | [Quick Start](#-quick-start) |[API Reference](#-api-reference)

</div>

## 🌟 Overview

**XingLing-Inference** is a lightweight yet powerful inference engine specifically built for the XingLing-0.68B [<sup>9</sup>](https://github.com/fancyovo/XingLing) model. 

The core philosophy of this project is to **implement everything from scratch** to master the low-level mechanics of the Transformer architecture and CUDA programming. Every performance-critical kernel is handcrafted without relying on heavyweight libraries like cuBLAS or cuDNN.

## 🚀 Key Features

-   **Handcrafted CUDA Kernels:** 
    -   Custom **GEMM (General Matrix Multiplication)** implementation with register-level blocking and optimization.
    -   **Multi-Head Attention** implementation inspired by the logic of FlashAttention-1.
-   **High-Performance CPU Backend:**
    -   Supports **AVX2 instruction set** for vectorization.
    -   Multi-threaded acceleration for efficient local execution.
-   **Advanced KV Cache Management:**
    -   Full **Prefill/Decode** phase separation.
    -   Optimized incremental updates during the decoding stage.
-   **Python Integration:** Seamlessly integrated with Python via **Pybind11**, allowing high-performance C++ execution within a familiar Python environment.

## 📊 Performance

*Tested on **NVIDIA RTX 3060 (12GB)**:*

| Phase       | Speed              | Description                          |
| :---------- | :----------------- | :----------------------------------- |
| **Prefill** | **3000+ tokens/s** | Parallel processing of input prompts |
| **Decode**  | **80+ tokens/s**   | Sequential token generation          |

## 🛠 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/fancyovo/XingLing-cpp.git
cd XingLing-cpp
```

### 2. Install Dependencies
Ensure you have a CUDA-capable environment and the following Python packages:
```bash
pip install torch transformers
```

### 3. Export Model Weights
Convert the trained PyTorch weights into the optimized binary format required by the C++ engine:
```bash
# You can customize MODEL_PATH and EXPORT_PATH in the script
python script/export_weights.py
```

### 4. Run Inference
```python
import torch    # Must import torch first to initialize CUDA context correctly
import llm_cpp_engine
from transformers import AutoTokenizer

# Configuration
TOKENIZER_PATH = "Qwen3-8B-Tokenizer"
MODEL_PATH = "data/model"

# Initialize
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
engine = llm_cpp_engine.InferenceEngine('cuda') # Use 'cpu' or 'cuda'
engine.load(MODEL_PATH)

print("Model loaded successfully!")

# Prepare Prompt
text = "Who is Albert Einstein?"
text_ids = tokenizer.encode(text)
input_ids = ( [engine.im_start_id, engine.role_id_user, engine.nl_id] + 
              text_ids + 
              [engine.im_end_id, engine.nl_id] +
              [engine.im_start_id, engine.role_id_assistant, engine.nl_id] )

# Prefill & First Token
token = engine.next_token(input_ids)
output_ids = input_ids + [token]

# Decoding Loop
for i in range(500):
    token = engine.next_token(token)
    if token == engine.im_end_id:
        break
    output_ids.append(token)

print(tokenizer.decode(output_ids))
```

## 📖 API Reference

### `llm_cpp_engine.InferenceEngine`

| Method                          | Description                                                  |
| :------------------------------ | :----------------------------------------------------------- |
| `InferenceEngine(device)`       | Constructor. `device` can be `'cpu'` or `'cuda'`.            |
| `load(path)`                    | Loads exported binary model weights from the specified path. |
| `next_token(tokens: list, ...)` | **Prefill Stage**: Resets KV cache, processes a list of tokens, and returns the next token. |
| `next_token(token: int, ...)`   | **Decode Stage**: Incremental KV cache update, processes a single token, and returns the next. |
| `to(device)`                    | Switches the engine between `'cpu'` and `'cuda'`.            |

**Generation Parameters:**
`next_token` supports the following sampling parameters:
- `temperature` (default: 0.7)
- `top_p` (default: 0.9)
- `repetition_penalty` (default: 1.1)

## 🏗 Model Configuration

| Parameter           | Value                    |
| :------------------ | :----------------------- |
| **Layers**          | 16                       |
| **Hidden Size**     | 1536                     |
| **FFN Dimension**   | 4096                     |
| **Attention Heads** | 24                       |
| **Max Context**     | 768                      |
| **Total Params**    | ~0.68B                   |
| **Tokenizer**       | Qwen3-8B (Vocab: 151669) |

## 🔗 Related Project
This inference engine is built for the **XingLing** model. For details on the training process, dataset, and model architecture, please visit the Main Training Repository [<sup>9</sup>](https://github.com/fancyovo/XingLing).

## ⚖️ License
This project is licensed under the Apache 2.0 License.

---
*If this project helped you understand CUDA or Transformers better, please give it a Star! ⭐️*

---