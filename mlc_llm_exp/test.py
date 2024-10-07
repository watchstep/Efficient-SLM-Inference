###  Import Library  ###
import os
import time
import random
import argparse

import numpy as np
import torch

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from mlc_llm import MLCEngine

###  Config  ###
parser = argparse.ArgumentParser(description="llama.cpp")
parser.add_argument("--model_path", type=str, default="/slm/")
parser.add_argument("--data_dir", type=str, default="/slm/")
parser.add_argument("--data_name", type=str, default="test_dataset.jsonl")
parser.add_argument('--seed',type=int, default=0)
parser.add_argument('--temperature', type=float, default=0.0)

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

# Create engine
# model = 
model = "HF://watchstep/Phi-3-medium-4k-instruct-fp16-MLC"
engine = MLCEngine(model)

###  Warm up ###
response = engine.chat.completions.create(
    messages=[{"role": "assistant", "content": "You are a helpful AI Assistant. Help users by replying to their queries and make sure the responses are polite. Do not hallucinate."}],
    model=model,
    stream=False)

for choice in response.choices:
    print(choice.message.content)

### Load data and Inference ### 
start = time.perf_counter()

data = load_dataset("json", data_files=os.path.join(config.data_dir, config.data_name))['train']
messages = data['message']

outs = []
for msg in messages:
    with torch.inference_mode(), torch.cuda.amp.autocast():
        response = engine.chat.completions.create(
            messages=msg,
            model=model,
            stream=False,
        )
    output = response.choices[0].message.content[0]

    outs.append([{
        'generated_text': output
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