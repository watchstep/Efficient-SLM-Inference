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

###  Config  ###
parser = argparse.ArgumentParser(description="llama.cpp")
parser.add_argument("--cache_dir", type=str, default=" ~/.cache/huggingface/hub/")
parser.add_argument("--data_dir", type=str, default="./data/")
parser.add_argument("--data_name", type=str, default="test_dataset.jsonl")
parser.add_argument('--seed',type=int, default=0)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--num_pred_tokens', type=int, default=10)

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
    torch.backends.cudnn.benchmark = True # False

_seed_everything(config.seed)

###  Load Model  ###
model = Llama.from_pretrained(
	repo_id="watchstep/Phi-3-medium-4k-instruct-fP32-gguf",
	filename="Phi-3-medium-4k-instruct-fp32.gguf",
	n_gpu_layers=30,
	n_ctx=1024,
	verbose=False,
    seed=config.seed,
	draft_model=LlamaPromptLookupDecoding(num_pred_tokens=num_pred_tokens),
    cache_dir=config.cache_dir,
)

def apply_chat_template(messages):
    formatted_messages = []
    
    for message in messages:
        formatted = [] 
        for msg in message:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "").strip()  
                
                if role == "user":
                    formatted.append(f"<|user|>\n\n{content}\n<|end|>\n")
        
        formatted.append("<|assistant|>\n")
        formatted_messages.append(''.join(formatted))
    
    return formatted_messages

###  Warm up ###
system_message = "You are a helpful AI Assistant. Help users by replying to their queries and make sure the responses are polite. Do not hallucinate."
prompt = f"<|assistant|>\n{system_message}<|end|>"

output = model(
      prompt,
      max_tokens=32,
      echo=False,
)

print(output['choices'][0]['text'])

### Load data and Inference ### 
start = time.perf_counter()

data = load_dataset("json", data_files="./data/test_dataset.jsonl")['train']
messages = data['message']
token_ids = apply_chat_template(messages)

outs = []
for token_id in token_ids:
    with torch.inference_mode(), torch.cuda.amp.autocast():
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