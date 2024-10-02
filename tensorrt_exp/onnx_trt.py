import os
import time

import numpy as np
import torch

import onnx
import onnxruntime as ort
from onnxruntime import InferenceSession

from optimum.onnxruntime import ORTModelForCausalLM

import warnings
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
import torch.backends.cudnn as cudnn

device = torch.device( "cuda" if torch.cuda.is_available() else cpu )

# 특정 경고 무시
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

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

# GPU 메모리 캐시 초기화
torch.cuda.empty_cache()

# CUDA 캐싱 할당기에서 할당된 모든 메모리 해제
torch.cuda.reset_peak_memory_stats()

model_path = './cuda-fp16/phi3-medium-4k-instruct-cuda-fp16.onnx'

model = onnx.load(model_path)
engine = backend.prepare(model, device='CUDA:0')
input_data = np.random.random(size=(32, 828)).astype(np.float32)
output_data = engine.run(input_data)[0]

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 15

onnx_model = ORTModelForCausalLM.load_model(model_path,
                                        session_options=sess_options,)