# MiniCPM-o 部署指南

> 📖 文档生成者：小阅 (Xiaoyue)  
> 📅 生成时间：2026-03-10

---

## 快速部署

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n minicpm python=3.10
conda activate minicpm

# 安装依赖
cd MiniCPM-o
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(torch.__version__)"
```

### 2. 模型下载

```bash
# HuggingFace
git lfs install
git clone https://huggingface.co/openbmb/MiniCPM-o-2_6

# 或 ModelScope
git clone https://modelscope.cn/openbmb/MiniCPM-o-2_6.git
```

### 3. 快速推理

```python
# 简单推理示例

from transformers import AutoModel, AutoTokenizer
from PIL import Image

# 加载模型
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-2_6",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    "openbmb/MiniCPM-o-2_6",
    trust_remote_code=True
)

# 加载图像
image = Image.open("test.jpg").convert('RGB')

# 构建对话
msgs = [{'role': 'user', 'content': '描述这张图片<image>'}]

# 推理
answer = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer
)

print(answer)
```

---

## 部署方式

### 1. Transformers 部署

```python
# 使用 Transformers 库

from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image

# 加载
processor = AutoProcessor.from_pretrained(
    "openbmb/MiniCPM-o-2_6",
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-2_6",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# 推理
image = Image.open("test.jpg")
question = "这张图片里有什么？"

inputs = processor(
    images=[image],
    text=question,
    return_tensors='pt'
).to('cuda')

outputs = model.generate(
    **inputs,
    max_new_tokens=512
)

answer = processor.decode(
    outputs[0],
    skip_special_tokens=True
)
```

### 2. vLLM 部署

```python
# vLLM 高性能推理

from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="openbmb/MiniCPM-o-2_6",
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
    max_model_len=2048
)

# 采样参数
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.9,
    max_tokens=512
)

# 推理
prompts = ["<image>\n描述这张图片"]

outputs = llm.generate(
    prompts,
    sampling_params,
    multi_modal_data={
        "image": Image.open("test.jpg")
    }
)

print(outputs[0].outputs[0].text)
```

**性能**:
- Batch=1: 20-25 tokens/s
- Batch=8: 80-100 tokens/s

### 3. llama.cpp 部署

```bash
# GGUF 量化推理

# 1. 下载 GGUF 模型
git lfs install
git clone https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf

# 2. 编译 llama.cpp
git clone https://github.com/OpenBMB/llama.cpp
cd llama.cpp
make -j

# 3. 推理
./main -m ../MiniCPM-o-2_6-gguf/minicpm-o-q4_k_m.gguf \
  -p "<image>\n描述这张图片" \
  --image test.jpg \
  -t 8 \
  -ngl 32 \
  -c 2048
```

**量化版本**:
- Q4_K_M: 5GB, 15 tokens/s (M1 Max)
- Q5_K_M: 6GB, 12 tokens/s
- Q8_0: 9GB, 8 tokens/s

### 4. Ollama 部署

```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. 拉取模型
ollama pull openbmb/minicpm-o-2_6

# 3. 运行
ollama run openbmb/minicpm-o-2_6 "描述这张图片" --image test.jpg

# 4. API 服务
ollama serve
```

**API 调用**:
```bash
curl http://localhost:11434/api/generate \
  -d '{
    "model": "openbmb/minicpm-o-2_6",
    "prompt": "描述这张图片",
    "images": ["base64_encoded_image"]
  }'
```

### 5. Web Demo 部署

```bash
# Gradio Web 界面

cd web_demos/minicpm-o_2.6

# 启动模型服务
python model_server.py &

# 启动 Web 界面
python chatbot_web_demo_o2.6.py
```

访问：http://localhost:7860

---

## 生产环境部署

### Docker 部署

```dockerfile
# Dockerfile

FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 安装 Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装依赖
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 复制代码
COPY . .

# 下载模型
RUN git lfs install && \
    git clone https://huggingface.co/openbmb/MiniCPM-o-2_6

EXPOSE 8000

CMD ["python3", "web_demos/minicpm-o_2.6/model_server.py"]
```

**构建和运行**:
```bash
docker build -t minicpm-o:latest .
docker run --gpus all -p 8000:8000 minicpm-o:latest
```

### Kubernetes 部署

```yaml
# deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: minicpm-o
spec:
  replicas: 2
  selector:
    matchLabels:
      app: minicpm-o
  template:
    metadata:
      labels:
        app: minicpm-o
    spec:
      containers:
      - name: minicpm-o
        image: minicpm-o:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
        env:
        - name: MODEL_PATH
          value: "/app/MiniCPM-o-2_6"
---
apiVersion: v1
kind: Service
metadata:
  name: minicpm-o-service
spec:
  selector:
    app: minicpm-o
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 多 GPU 部署

### 模型并行

```python
# 多 GPU 推理

import torch
from transformers import AutoModel

# 自动分配
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-2_6",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # 自动分配到多个 GPU
)

# 或手动指定
device_map = {
    "vision_tower": 0,      # 视觉编码器在 GPU 0
    "resampler": 0,         # Resampler 在 GPU 0
    "llm": 1,               # LLM 在 GPU 1
}

model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-2_6",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map=device_map
)
```

### DeepSpeed 部署

```python
# deepspeed_config.json

{
  "tensor_parallel": {
    "tp_size": 2
  },
  "dtype": "bfloat16",
  "replace_method": "auto"
}
```

**运行**:
```bash
deepspeed --num_gpus=2 inference.py \
  --deepspeed deepspeed_config.json
```

---

## 性能优化

### 1. 量化

```python
# 4bit 量化

from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-2_6-int4",
    quantization_config=quant_config,
    device_map="auto"
)
```

**效果**:
- 显存：18GB → 5GB
- 速度：提升 30-50%

### 2. Flash Attention

```python
# 启用 Flash Attention 2

model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-2_6",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

**要求**:
- torch >= 2.1.0
- flash-attn >= 2.3.0

### 3. 批处理

```python
# 批处理推理

from vllm import LLM

llm = LLM(
    model="openbmb/MiniCPM-o-2_6",
    max_num_batched_tokens=8192,
    max_num_seqs=16
)

# 批处理生成
prompts = [
    "<image>\n描述图片 1",
    "<image>\n描述图片 2",
    # ...
]

outputs = llm.generate(prompts, sampling_params)
```

---

## 故障排查

### 常见问题

#### 1. 显存不足

**错误**: `CUDA out of memory`

**解决**:
```bash
# 使用量化版本
git clone https://huggingface.co/openbmb/MiniCPM-o-2_6-int4

# 或减少 batch size
export VLLM_GPU_MEMORY_UTILIZATION=0.7
```

#### 2. 模型加载失败

**错误**: `model loading failed`

**解决**:
```bash
# 重新下载模型
rm -rf MiniCPM-o-2_6
git lfs install
git clone https://huggingface.co/openbmb/MiniCPM-o-2_6

# 检查文件完整性
ls -lh MiniCPM-o-2_6/*.safetensors
```

#### 3. 推理速度慢

**优化**:
```bash
# 使用 vLLM
pip install vllm

# 或 llama.cpp
git clone https://github.com/OpenBMB/llama.cpp
```

---

_部署指南文档结束_
