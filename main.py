import torch
import llm_cpp_engine
from transformers import AutoTokenizer
import os

TOKENIZER_PATH = "Qwen3-8B-Tokenizer"
MODEL_PATH = "data/model"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
engine = llm_cpp_engine.InferenceEngine('cuda')
engine.load(MODEL_PATH)

print("model loaded successfully")

text = "天空是什么颜色？"
text_ids = tokenizer.encode(text)
input_ids = (    [engine.im_start_id, engine.role_id_user, engine.nl_id] + 
                text_ids + 
                [engine.im_end_id, engine.nl_id] +
                [engine.im_start_id, engine.role_id_assistant, engine.nl_id])
# input_ids = text_ids
token = engine.next_token(input_ids)
output_ids = input_ids + [token]


for i in range(500):
    token = engine.next_token(token)
    if token == engine.im_end_id:
        break
    output_ids.append(token)
    print(i, token)

output_text = tokenizer.decode(output_ids)
print(output_text)