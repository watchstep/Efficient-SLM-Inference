import os
import time
import random
import argparse

import numpy as np
import torch


from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
import torch_tensorrt

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

def compile(model):
    trt_model = torch.compile(model, 
                        backend="torch_tensorrt",
                         options={
                            "truncate_long_and_double": True,
                            "precision": torch.half,
                            "min_block_size": 1,
                            "workspace_size": 1 << 33,
                            "debug": False,
                            "verbose":True
                        }, 
                        dynamic=False)
    return trt_model

def generate(model, tokenizer, data_file_path):
    generation_args = {
        "max_new_tokens": 200,
        "temperature": 0.0,
        "do_sample": False,
    }

    system_message = "You are a helpful AI Assistant. Help users by replying to their queries and make sure the responses are polite. Do not hallucinate."
    PROMPT = f"<|system|>\n{system_message}<|end|>"

    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # Load the dataset
    data = load_dataset("json", data_files=data_file_path)['train']
    messages = data["message"]
    
    toekn_ids = tokenizer.apply_chat_template(messages, 
                                            add_generation_prompt=True, 
                                            tokenize=False,)
    # Add the system prompt
    toekn_ids.insert(0, PROMPT)
    
    # Tokenize the prompts
    inputs = tokenizer(toekn_ids, return_tensors="pt", padding=True)
    inputs = {k: v.type(torch.int32).to(DEVICE) for k, v in inputs.items()}
    
    with torch.inference_mode():
        outs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **generation_args
        )
        
    generated_texts = tokenizer.batch_decode(outs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Ouput in JSON format
    results = [
        [{'generated_text': text.rpartition('\n')[2]}]
        for text in generated_texts[1:]
    ]
    
    torch.cuda.synchronize()
    end = time.perf_counter()

    print("===== Answers =====")
    correct = 0
    for i, out in enumerate(results):
        correct_answer = data[i]["answer"]
        answer = out[0]["generated_text"].lstrip().replace("\n","")
        if answer == correct_answer:
            correct += 1
        print(answer)
    
    print("===== Perf result =====")
    print("Elapsed_time: ", end-start)
    print(f"Correctness: {correct}/{len(data)}")

    return results, end-start

def main(config):
    model_id = os.path.join(config.model_path, config.model_name)

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                device_map="auto",
                                                torch_dtype="auto",
                                                trust_remote_code=True,
                                                torchscript=True,
                                                attn_implementation="eager")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    trt_model = compile(model)
    data_file_path = os.path.join(config.data_path, config.data_name)

    results, elapsed_time = generate(model=trt_model,
                                    tokenizer=tokenizer, 
                                    data_file_path=data_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TensorRT")
    parser.add_argument("--model_path", type=str, default="models/", 
                        help="Path to the directory with HF model files")
    parser.add_argument("--model_name", type=str, default="Phi-3-medium-4k-instruct")
    parser.add_argument("--data_path", type=str, default="data/",
                        help="Path to the directory with data files")
    parser.add_argument("--data_name", type=str, default="test_dataset.jsonl")
    parser.add_argument('--seed',type=int, default=0)
    
    config = parser.parse_args()
    _seed_everything(config.seed)
    main(config)

