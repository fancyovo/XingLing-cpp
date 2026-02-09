from transformers import AutoTokenizer

TOKENIZER_PATH = "Qwen3-8B-Tokenizer"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

im_start_id = tokenizer.convert_tokens_to_ids(IM_START)
im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
nl_ids = tokenizer.encode("\n", add_special_tokens=False)
user_role_ids = tokenizer.encode("user", add_special_tokens=False)
assist_role_ids = tokenizer.encode("assistant", add_special_tokens=False)

print(f"im_start_id: {im_start_id}")
print(f"im_end_id: {im_end_id}")
print(f"nl_ids: {nl_ids}")
print(f"user_role_ids: {user_role_ids}")
print(f"assistant_role_ids: {assist_role_ids}")