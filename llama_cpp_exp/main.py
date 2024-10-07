###  Import Library  ###
import os
import time
import random
import argparse

import numpy as np
import torch

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
import json

###  Config  ###
parser = argparse.ArgumentParser(description="llama.cpp")
parser.add_argument("--cache_dir", type=str, default=" ~/.cache/huggingface/hub/")
parser.add_argument("--data_dir", type=str, default="./data/")
parser.add_argument("--data_name", type=str, default="test_dataset.jsonl")
parser.add_argument('--seed',type=int, default=0)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--num_pred_tokens', type=int, default=10)
parser.add_argument('-n_gpu_layers, type=int', default=-1)

config = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

_seed_everything(config.seed)

###  Load Model  ###
model = Llama.from_pretrained(
	repo_id="watchstep/Phi-3-medium-4k-instruct-fp32-gguf",
	filename="Phi-3-medium-4k-instruct-fp32.gguf",
	n_gpu_layers=-1, #In this model, we use 41 layers
	n_ctx=1024, # Context length or window size for the model's token processing.
	n_batch=1024, # Number of tokens processed per batch.
	verbose=False,
    seed=config.seed,
	draft_model=LlamaPromptLookupDecoding(num_pred_tokens=config.num_pred_tokens), 
    # Use speculative decoding for faster inference, predicting 10 tokens ahead.
)

### Prompt Setting ### 
system_message="You are a highly helpful and knowledgeable AI assistant specializing in answering user queries accurately and politely."               
prompt = f"<|system|>\n{system_message}\n"

def apply_chat_template(messages):
    formatted_messages = []
    
    formatted_messages.extend(
        ''.join(
            [f"{prompt}<|user|>\n{msg.get('content', '').strip()}\n<|end|>\n\n" if msg.get('role') == 'user' else '' for msg in message]
        ) + "<|assistant|>\n" for message in messages
    )
    
    return formatted_messages

# Load the dataset in JSON format, specifying the file location for training data.
data = load_dataset("json", data_files="./data/test_dataset.jsonl")['train']
messages = data['message']
token_ids = apply_chat_template(messages)

###  Warm up ###
dummy = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

def save_to_jsonl(token_ids, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for token_id in token_ids:
            json.dump({"token_id": token_id}, file)
            file.write('\n')  

def load_from_jsonl(filename):
    token_ids = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            token_ids.append(data['token_id'])  
    return token_ids

save_to_jsonl(token_ids,"processed_data.jsonl")

dummy = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

output = model.create_chat_completion(
      messages=dummy
)

### Load data and Inference ### 
start = time.perf_counter()
token_ids = load_from_jsonl("processed_data.jsonl")
outs = []
for token_id in token_ids:
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        output = model(token_id,
        temperature=config.temperature, # Temperature for sampling to control randomness.
        echo=False)

    out = output['choices'][0]['text']

    outs.append([{
        'generated_text': out
    }])

end = time.perf_counter()

#### Benchmark ###
print("===== Answers =====")
correct = 0

for i, out in enumerate(outs):
    correct_answer = data[i]["answer"]
    answer = out[0]["generated_text"].lstrip().replace("\n","")
    if answer == correct_answer:
        correct += 1
 
print("===== Perf result =====")
print("Elapsed_time: ", end-start)
print(f"Correctness: {correct}/{len(data)}")