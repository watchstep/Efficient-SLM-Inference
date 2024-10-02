import os
import time

import numpy as np
import torch
import argparse

from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer, 
                          Phi3ForCausalLM,
                          Phi3Config,
                          pipeline,
                          )
from transformers.pipelines.pt_utils import KeyDataset
from huggingface_hub import hf_hub_download
import torch_tensorrt

import utils

parser = argparse.ArgumentParser(description="TensorRT")
parser.add_argument("--model_path", type=str, default="models/")
parser.add_argument("--model_name", type=str, default="Phi-3-medium-4k-instruct")
parser.add_argument("--data_path", type=str, default="data/")
parser.add_argument("--dataset", type=str, default="test_dataset.jsonl")
parser.add_argument('--seed',type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument('--dtype',type=str, default="auto")
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--temperature', type=float, default=0.0)

config = parser.parse_args([])
utils.seed_everything(config.seed)
CURR_PATH = os.getcwd()

model_id = os.path.join(CURR_PATH, config.model_path, config.model_name)

model = Phi3ForCausalLM.from_pretrained(
    model_id,
    device_map="auto", # "cuda"
    torch_dtype="auto",
    trust_remote_code=True,
    torchscript=True
    # attn_implementation="flash_attention_2" # Running flash attention
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

enabled_precisions = {torch.half} # Run with FP16
debug = True
# workspace_size = 20 << 30
min_block_size = 7  # Lower value allows more graph segmentation
torch_executed_ops = {}

compilation_args = {
    "enabled_precisions": enabled_precisions,
    "debug": debug,
    # "workspace_size": workspace_size,
    "min_block_size": min_block_size,
    "torch_executed_ops": torch_executed_ops,
}


trt_model = torch.compile(
                        model, 
                        # mode="max-autotune", 
                        backend="torch_tensorrt",
                        dynamic=False,
                        fullgraph=True,
                        options=compilation_args)

pipe = pipeline(
    "text-generation",
    model=trt_model,
    tokenizer=tokenizer,
    batch_size=config.batch_size,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

start = time.time()
data = load_dataset("json", data_files="./data/test_dataset.jsonl")['train']
outs = pipe(KeyDataset(data, 'message'), **generation_args)
end = time.time()


print("===== Answers =====")
correct = 0
for i, out in enumerate(outs):
    correct_answer = data[i]["answer"]
    answer = out[0]["generated_text"].lstrip().replace("\n","")
    if answer == correct_answer:
        correct += 1
    # print(answer)
 
print("===== Perf result =====")
print("Elapsed_time: ", end-start)
print(f"Correctness: {correct}/{len(data)}")