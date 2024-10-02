import os
import time
import random
import argparse

import numpy as np
import torch


from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset

import onnxruntime_genai as og

parser = argparse.ArgumentParser(description="ONNXRUNTIME")
parser.add_argument("--model_dir", type=str, default="./cuda-fp16/",)
parser.add_argument("--data_dir", type=str, default="data/",)
parser.add_argument('--data_name', type=str, default="test_dataset.jsonl")

config = parser.parse_args()
model = og.Model(config.model_dir)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

search_options = {"max_length": 1046,"temperature":0.0}
params = og.GeneratorParams(model)
params.set_search_options(**search_options)

start = time.perf_counter()

hf_tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
data = load_dataset("json", data_files=os.path.join(config.data_dir, config.data_name))['train']
messages = data['message']

token_ids = hf_tokenizer.apply_chat_template(messages, 
                                        add_generation_prompt=True, 
                                        tokenize=False,)

system_message = "You are a helpful AI Assistant. Help users by replying to their queries and make sure the responses are polite. Do not hallucinate."
PROMPT = f"<|system|>\n{system_message}<|end|>"                                        
token_ids.insert(0, PROMPT)

outs = []
for token_id in token_ids[1:]:
    input_tokens = tokenizer.encode(token_id)
    params = og.GeneratorParams(model)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)

    text = ''
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        
        new_token = generator.get_next_tokens()[0]
        text += tokenizer_stream.decode(new_token)
    
    outs.append(
        [{'generated_text': text}]
        )

end = time.perf_counter()

print("===== Answers =====")
correct = 0
for i, out in enumerate(outs):
    correct_answer = data[i]["answer"]
    answer = out[0]["generated_text"].lstrip().replace("\n","")

    # print(f"Correct Answer: {correct_answer}")
    # print(f"Generated Answer: {answer}")
    if answer == correct_answer:
        correct += 1
    
print("===== Perf result =====")
print("Elapsed_time: ", end-start)
print(f"Correctness: {correct}/{len(data)}")