# MiniCPM-o API 参考手册

> 📖 文档生成者：小阅 (Xiaoyue)  
> 📅 生成时间：2026-03-10

---

## API 概览

### 端点

| 服务 | 端点 | 端口 |
|------|------|------|
| HTTP API | `/api/generate` | 11434 (Ollama) |
| HTTP API | `/generate` | 8000 (vLLM) |
| WebSocket | `/ws/chat` | 7860 (Gradio) |

---

## Ollama API

### 生成接口

**端点**: `POST /api/generate`

**请求**:
```json
{
  "model": "openbmb/minicpm-o-2_6",
  "prompt": "描述这张图片",
  "images": ["base64_encoded_image"],
  "stream": false,
  "options": {
    "temperature": 0.6,
    "top_p": 0.9,
    "num_predict": 512
  }
}
```

**响应**:
```json
{
  "model": "openbmb/minicpm-o-2_6",
  "response": "这张图片展示了...",
  "done": true,
  "total_duration": 1234567890,
  "load_duration": 123456789,
  "eval_count": 100,
  "eval_duration": 987654321
}
```

**cURL 示例**:
```bash
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/minicpm-o-2_6",
    "prompt": "描述这张图片",
    "images": ["iVBORw0KG..."],
    "stream": false
  }'
```

### 流式接口

**端点**: `POST /api/generate` (stream=true)

**请求**:
```json
{
  "model": "openbmb/minicpm-o-2_6",
  "prompt": "描述这张图片",
  "images": ["..."],
  "stream": true
}
```

**响应** (多行):
```json
{"response":"这","done":false}
{"response":"张","done":false}
{"response":"图","done":false}
{"response":"片","done":false}
...
{"response":"。","done":true}
```

---

## vLLM API

### 生成接口

**端点**: `POST /generate`

**请求**:
```json
{
  "model": "openbmb/MiniCPM-o-2_6",
  "prompt": "<image>\n描述这张图片",
  "max_tokens": 512,
  "temperature": 0.6,
  "top_p": 0.9,
  "stream": false
}
```

**响应**:
```json
{
  "id": "cmpl-123456",
  "object": "text_completion",
  "created": 1234567890,
  "model": "openbmb/MiniCPM-o-2_6",
  "choices": [{
    "index": 0,
    "text": "这张图片展示了...",
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 200,
    "total_tokens": 300
  }
}
```

**Python 示例**:
```python
import requests
import base64

# 编码图像
with open("test.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# 请求
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "model": "openbmb/MiniCPM-o-2_6",
        "prompt": "<image>\n描述这张图片",
        "multi_modal_data": {
            "image": image_b64
        },
        "max_tokens": 512
    }
)

print(response.json()["choices"][0]["text"])
```

---

## OpenAI 兼容 API

### Chat Completions

**端点**: `POST /v1/chat/completions`

**请求**:
```json
{
  "model": "openbmb/minicpm-o-2_6",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "描述这张图片"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,..."
          }
        }
      ]
    }
  ],
  "max_tokens": 512,
  "temperature": 0.6
}
```

**响应**:
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "openbmb/minicpm-o-2_6",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "这张图片展示了..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 200,
    "total_tokens": 300
  }
}
```

---

## Python SDK

### 安装

```bash
pip install openai
```

### 使用示例

```python
from openai import OpenAI
import base64

# 初始化客户端
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 编码图像
with open("test.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# 对话
response = client.chat.completions.create(
    model="openbmb/minicpm-o-2_6",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图片"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }
    ],
    max_tokens=512
)

print(response.choices[0].message.content)
```

---

## 错误码

| 错误码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求格式错误 |
| 401 | 认证失败 |
| 404 | 模型不存在 |
| 429 | 请求过多 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 |

**错误响应**:
```json
{
  "error": {
    "code": "invalid_request_error",
    "message": "图像格式不支持",
    "type": "invalid_request_error"
  }
}
```

---

_API 参考手册结束_
