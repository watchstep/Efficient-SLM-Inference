## Phi-3-medium-4k-instruct Inference on NVIDIA Jetson AGX Orin

### 1. Introduction

The **Computer Engineering Challenge** is focused on optimizing the performance of LLMs, with an emphasis on reducing inference latency while maintaining high accuracy on given datasets. 
We utilized the **Phi3-medium-4k-instruct** model, a 14-billion parameter LLM, to tackle a question-answering task based on a structured dataset. To address the challenge of reducing inference latency, we implemented several key optimizations:

- **Model Conversion**: We converted **hf** into the **gguf** format, enabling more efficient loading and execution during inference.
- **Speculative Decoding  (Leviathan et al., 2023; Chen et al., 2023a)** : This technique leverages a draft model that predicts a few tokens ahead (in our case, 10 tokens), allowing for parallel token generation and reducing the number of iterations required for generating the final output.
    
    ![image.png](https://github.com/user-attachments/assets/841422c2-1b48-4101-bec2-40bbb089dd06)
    
    - **Speculative Decoding** extends the **Draft-then-Verify** paradigm by incorporating various sampling techniques. This uses pre-trained smaller models, eliminating the need for extra training and simplifying deployment.
- **Dataset Handling**: We streamlined the process of loading the question-answering (QA) dataset, ensuring efficient data flow during inference.
- **Prompt Optimization**: By refining and optimizing the prompts used during inference, we further improved the response accuracy and speed of the model.

### 2. Set up

### 2-1. Install Docker

[hub.docker.com](https://hub.docker.com/r/0914eagle/samsung)

```bash
sudo docker pull samsung
sudo docker run --runtime nvidia -it --name {컨테이너 이름} 
-v  samsung:latest /bin/bash

sudo docker start {도커 컨테이너 이름}
sudo docker attach {도커 컨테이너 이름}
```

- **Prerequisites**
    
    ```bash
    CUDACXX=/usr/local/cuda-12.2/bin/nvcc CMAKE_ARGS=
    "-DGGML_CUDA=on" 
    FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall 
    --upgrade --no-cache-dir --verbose
    
    python -c "import llama_cpp" 
    # Check if 'llama-cpp-python' imports successfully
    ```
    

### 2-2. Download Model

❓ How to Convert from `HF` to `GGUF`? Please refer to 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zdN3JvgoZzJWFdFm4mY0tmkdOzhablHi?usp=sharing)

Download GGUF
[![Huggingface space](https://img.shields.io/badge/🤗-Phi3_medium_4k_instruct_fp32_gguf%20-yellow.svg)](https://huggingface.co/watchstep/Phi-3-medium-4k-instruct-fp32-gguf)


GGUF format for the [microsoft](https://huggingface.co/microsoft)/[Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) **(not quantization)**

- `Phi-3-medium-4k-instruct-fp32.gguf` : 14B parameters, FP32 (32-bit floating point) precision (*We recommend using this file.)
- `Phi-3-medium-4k-instruct-fp16.gguf` : 14B parameters, FP16 (16-bit floating point) precision

```bash
git lfs install
git-lfs clone https://huggingface.co/watchstep/Phi-3-medium-4k-
instruct-fp32-gguf
```

## 3. Options

❗️(**required**) You should update this path to point to the location where your model is stored.

If you want to use fp32 choose “`"Phi-3-medium-4k-instruct-fp32-gguf"`", else `"Phi-3-medium-4k-instruct-fp16.gguf"`

`--model_dir`: **Type**: `str`, **Default**: `"Phi-3-medium-4k-instruct-fp32.gguf"`

❗️(**required**) Modify this to point to the correct dataset that you mount to local folder.

`--data_path`: **Type**: `str`, **Default**: `"new_test_dataset.jsonl"`

---

1. **❗️`--model_dir`** (string, required=True)
    - The directory where the model files are stored. This should be the folder path where the model files are located.
2. **`--model_name`** (string, default: `"Phi-3-medium-4k-instruct-fp32.gguf"`)
    - The name of the model file you want to use for inference. The default model is set to `Phi-3-medium-4k-instruct-fp32.gguf`. Make sure that the model file is present in the `model_dir`.
3. **❗️`--data_path`** **(string, required=True)**
    - The path where the input dataset file is stored. Thedata file should contain the test data in JSON format for inference.
4. **`--seed`** (integer, default: `0`)
    - The seed value for reproducibility. Change this if you want to control randomness for different runs.
5. **`--temperature`** (float, default: `0.0`)
    - The temperature to use for sampling.
6. **`--verbose`** (bool, default: `False` )
    - Whether to enable verbose logging during execution.
7. **`--n_gpu_layers`** (integer, default: `-1`)
    - Number of layers to offload to GPU for processing
    If -1, all layers are offloaded.
8. **`--num_pred_tokens`** (integer, default: `10`)
    - The number of tokens the model will predict during inference for speculative decoding.

## 4. Run

```bash
cd /

python final.py --model_dir "/path/to/model" \
--data_path "/path/to/data" 

python final.py --model_dir 
"./Phi-3-medium-4k-instruct-fp32-gguf" --data_path
 "./data/test_dataset.jsonl"
```

**✅ NOTE**

![image.png](https://github.com/user-attachments/assets/6c18513b-0960-40d2-b07a-f8e30a0a5392)

 Be patient😆. It takes about 2 hours to offload the model parameters for each layer onto the GPU in Jetson Orin AGX 32GB. (You can view the logging by setting `verbose=True`)
