###  Import Library  ###
import os
import time
import random
import argparse
import json

import numpy as np
import torch

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

###  Config  ###
parser = argparse.ArgumentParser(description="llama.cpp")
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--model_name", type=str, default="Phi-3-medium-4k-instruct-fp32.gguf")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument('--seed',type=int, default=0)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--verbose', type=bool, default=False)
parser.add_argument('--n_gpu_layers', type=int, default=-1)
parser.add_argument('--num_pred_tokens', type=int, default=10)

config = parser.parse_args()

def _seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # False

_seed_everything(config.seed)

###  Load Model  ###
model = Llama(
	model_path=os.path.join(config.model_dir, config.model_name),
	n_gpu_layers=config.n_gpu_layers,
	n_ctx=1024,
	n_batch=1024,
	verbose=config.verbose,
    seed=config.seed,
	draft_model=LlamaPromptLookupDecoding(num_pred_tokens=config.num_pred_tokens),
)

###  Prompt  ###
system_message = "You are a highly helpful and knowledgeable AI assistant specializing in answering user queries accurately and politely."
prompt = f"<|system|>\n{system_message}\n"

def apply_chat_template(messages):
    formatted_messages = []
    
    formatted_messages.extend(
        ''.join(
            [f"{prompt}<|user|>\n{msg.get('content', '').strip()}\n<|end|>\n\n" if msg.get('role') == 'user' else '' for msg in message]
        ) + "<|assistant|>\n" for message in messages
    )
    
    return formatted_messages


###  Process data  ###
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

data = load_dataset("json", data_files=config.data_path)['train']
messages = data['message']
token_ids = apply_chat_template(messages)
save_to_jsonl(token_ids,"processed_data.jsonl")

###  Warm up ###
dummy = "Can you provide ways to eat combinations of bananas and dragonfruits?"

output = model(
      dummy,
      max_tokens=32,
      echo=False,
)

print(output['choices'][0]['text'])


### Load data and Inference ### 
start = time.perf_counter()
token_ids = load_from_jsonl("processed_data.jsonl")

outs = []
for token_id in token_ids:
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        output = model(token_id,
        temperature=config.temperature,
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