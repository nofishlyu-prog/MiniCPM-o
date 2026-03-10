# MiniCPM-o 模型架构详解

> 📖 文档生成者：小阅 (Xiaoyue)  
> 📅 生成时间：2026-03-10  
> 🎯 模型：MiniCPM-o 4.5 (9B)

---

## 📑 目录

1. [模型概览](#模型概览)
2. [整体架构](#整体架构)
3. [视觉编码器](#视觉编码器)
4. [Resampler](#resampler)
5. [LLM 骨干](#llm-骨干)
6. [多模态融合](#多模态融合)
7. [训练方法](#训练方法)
8. [推理优化](#推理优化)

---

## 模型概览

### MiniCPM-o 4.5 参数配置

| 组件 | 模型 | 参数量 |
|------|------|--------|
| Vision Encoder | EVA-02 Enormous | 1.1B |
| Audio Encoder | Whisper-medium | 769M |
| LLM Backbone | Qwen2.5-8B | 8B |
| **总计** | | **~9B** |

### 技术特点

- **全双工 multimodal live streaming** - 同时看、听、说
- **Gemini 2.5 Flash 级别** - 视觉/语音能力
- **64 token 图像表示** - 高效压缩
- **30+ 语言支持** - 多语言能力

---

## 整体架构

```
输入层:
┌─────────────────────────────────────────┐
│  图像 (224x224) / 视频 / 音频 / 文本     │
└────────────┬────────────────────────────┘
             │
    ┌────────┼────────┬──────────┐
    │        │        │          │
    ▼        ▼        ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ │
│ EVA-02 │ │Whisper │ │ Text   │ │
│ Vision │ │ Audio  │ │ Embed  │ │
│ Encoder│ │ Encoder│ │        │ │
└───┬────┘ └───┬────┘ └───┬────┘ │
    │          │          │      │
    │ 256x1792 │ 音频特征  │      │
    │          │          │      │
    ▼          ▼          │      │
┌─────────────────────┐   │      │
│   Resampler + Proj  │   │      │
│   (特征对齐 + 投影)    │   │      │
└──────────┬──────────┘   │      │
           │              │      │
           │ 64x4096      │      │
           │              │      │
           ▼              ▼      ▼
        ┌──────────────────────┐
        │  LLM (Qwen2.5-8B)    │
        │                      │
        │  - Self-Attention    │
        │  - Cross-Attention   │
        │  - FFN               │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────┐
        │   Output Head    │
        │  - Text Logits   │
        │  - Speech (TTS)  │
        └──────────────────┘
```

---

## 视觉编码器

### EVA-02 Enormous

**架构**: Vision Transformer (ViT)

```python
# timm.create_model('eva02_enormous_patch14_clip_224.laion2b_plus')

class EVA02(VisionTransformer):
    """
    EVA-02 视觉编码器
    
    输入：224x224 图像
    输出：256 个视觉 token (14x14 patch + 2x2 下采样)
    """
    
    # 关键参数
    patch_size = 14          # patch 大小
    embed_dim = 1792         # 嵌入维度
    depth = 64               # Transformer 层数
    num_heads = 16           # 注意力头数
    mlp_ratio = 8.5714       # MLP 比例
```

### 特征提取流程

```
1. 图像分块
   输入：(3, 224, 224)
   Patch: 14x14
   输出：(256, 14*14=196 patches)
   
2. Position Embedding
   添加 2D 正弦位置编码
   
3. Transformer 编码 (64 层)
   每层：
   - Multi-Head Self-Attention
   - LayerNorm
   - MLP (GELU 激活)
   
4. 输出
   形状：(256, 1792)
   256 个视觉 token，每个 1792 维
```

### 使用倒数第二层

```python
# omnilmm/model/omnilmm.py

# 使用倒数第二层输出 (而非最后一层)
vision_tower.blocks[-1] = Identity()

# 原因：
# 1. 最后一层过于抽象，丢失细节
# 2. 倒数第二层保留更多空间信息
# 3. 实验证明效果更好
```

---

## Resampler

### 架构设计

**文件**: `omnilmm/model/resampler.py`

```python
class Resampler(nn.Module):
    """
    2D Perceiver Resampler
    
    功能：将 N 个视觉 token 压缩为 64 个查询 token
    
    输入：(N, 1792) - 视觉特征
    输出：(64, 4096) - 压缩后的特征
    """
    
    def __init__(
        self,
        grid_size=8,           # 8x8 = 64 queries
        embed_dim=4096,        # LLM 维度
        num_heads=32,          # 注意力头数
        kv_dim=1792,           # Vision 维度
    ):
        super().__init__()
        
        # 可学习查询 (64 个)
        self.num_queries = grid_size ** 2  # 64
        self.query = nn.Parameter(
            torch.zeros(self.num_queries, embed_dim)
        )
        trunc_normal_(self.query, std=.02)
        
        # 2D 位置编码
        self.pos_embed = nn.Parameter(
            torch.from_numpy(
                get_2d_sincos_pos_embed(embed_dim, grid_size)
            ).float()
        ).requires_grad_(False)
        
        # KV 投影
        self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        
        # Cross-Attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # LayerNorm
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.ln_post = norm_layer(embed_dim)
        
        # 输出投影
        self.proj = nn.Parameter(
            (embed_dim ** -0.5) * torch.randn(embed_dim, embed_dim)
        )
```

### 前向传播

```python
def forward(self, x, attn_mask=None):
    """
    Resampler 前向
    
    Args:
        x: 视觉特征 (N, kv_dim)
    
    Returns:
        out: 重采样特征 (64, embed_dim)
    """
    # 1. KV 投影
    x = self.kv_proj(x)  # (N, embed_dim)
    x = self.ln_kv(x)
    
    # 2. 添加位置编码
    pos_embed = self.pos_embed  # (64, embed_dim)
    
    # 3. Cross-Attention
    # Query: 可学习查询 + 位置编码
    q = self.query + pos_embed
    q = self.ln_q(q)
    
    # Key, Value: 视觉特征
    k, v = x.unsqueeze(1), x.unsqueeze(1)
    
    # Attention
    out, _ = self.attn(
        q.unsqueeze(1),  # (64, 1, embed_dim)
        k, v,
        key_padding_mask=attn_mask
    )
    
    # 4. 后处理
    out = out.squeeze(1)  # (64, embed_dim)
    out = self.ln_post(out)
    out = out @ self.proj  # 投影
    
    return out  # (64, embed_dim)
```

### 为什么用 Resampler？

| 方法 | Token 数 | 信息损失 | 计算效率 |
|------|---------|---------|---------|
| 直接池化 | 64 | 高 | 高 |
| Perceiver Resampler | 64 | 中 | 中 |
| **2D Resampler** | **64** | **低** | **高** |

**优势**:
- 2D 位置编码保留空间结构
- 可学习查询主动提取信息
- Cross-Attention 高效融合

---

## LLM 骨干

### Qwen2.5-8B

**架构**: Transformer Decoder-only

```python
# Transformers 加载
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

**关键参数**:

| 参数 | 值 |
|------|-----|
| 词表大小 | 151,936 |
| 隐藏层维度 | 4096 |
| 注意力头数 | 32 |
| 层数 | 32 |
| 中间层维度 | 22016 |
| 上下文长度 | 32K |

### Mistral 版本

早期版本使用 Mistral-7B:

```python
class OmniLMMModel(MistralModel):
    """继承自 Mistral"""
    
    config_class = OmniLMMConfig
    model_type = "omnilmm"
```

---

## 多模态融合

### 特征拼接策略

```python
def get_vllm_embedding(self, data):
    """
    多模态特征融合
    
    策略：在文本嵌入中插入视觉特征
    """
    # 1. 获取视觉特征
    vision_hidden_states = [
        self.get_vision_embedding(pv.unsqueeze(0))[0]
        for pv in pixel_values_list
    ]  # [(64, 4096), ...]
    
    # 2. 文本嵌入
    inputs_embeds = self.embed_tokens(data['input_ids'])
    
    # 3. 融合
    new_input_embeds = []
    
    for cur_input_ids, cur_input_embeds in zip(
        data['input_ids'], 
        inputs_embeds
    ):
        # 找到<im_start>位置
        image_start_tokens = torch.where(
            cur_input_ids == self.vision_config.im_start_token
        )[0]
        
        for image_start_token_pos in image_start_tokens:
            # 提取视觉特征
            cur_image_features = vision_hidden_states[cur_image_idx]
            
            # 拼接：[前缀] + [视觉] + [后缀]
            cur_new_input_embeds = torch.cat(
                (
                    cur_input_embeds[:image_start_token_pos+1],
                    cur_image_features,  # 64 个视觉 token
                    cur_input_embeds[image_start_token_pos + 64 + 1:]
                ),
                dim=0
            )
        
        new_input_embeds.append(cur_new_input_embeds)
    
    inputs_embeds = torch.stack(new_input_embeds, dim=0)
    
    return inputs_embeds, vision_hidden_states
```

### Token 布局

```
输入序列:
[<im_start>] + [<im_patch>×64] + [<im_end>] + [文本]

展开:
位置 0:    <im_start>        (1 token)
位置 1-64: <im_patch>×64     (64 tokens, 视觉特征)
位置 65:   <im_end>          (1 token)
位置 66+:  文本 token

总视觉 token 数：66 个
```

---

## 训练方法

### 两阶段训练

#### 阶段 1: 预训练 (LMM)

**目标**: 对齐视觉和语言模态

```
数据：LLaVA-665K, COCO, VQAv2
Batch Size: 256
Learning Rate: 1e-4
Epochs: 1
Freeze: LLM (只训练 Vision + Resampler)
```

#### 阶段 2: 指令微调 (SFT)

**目标**: 遵循指令能力

```
数据：多模态指令数据
Batch Size: 64
Learning Rate: 2e-5
Epochs: 3
Freeze: 无 (全量微调)
```

### 损失函数

```python
# 标准 Causal LM Loss

class CausalLMOutputWithPast:
    loss: Optional[torch.FloatTensor]
    
    # 计算
    loss_fct = CrossEntropyLoss()
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = loss_fct(
        shift_logits.view(-1, config.vocab_size),
        shift_labels.view(-1)
    )
```

### 数据增强

```python
# omnilmm/model/utils.py

def build_transform(is_train, input_size, std_mode='OPENAI_CLIP'):
    """
    图像变换
    
    训练时:
    - RandomResizedCrop
    - RandomHorizontalFlip
    - ColorJitter
    
    推理时:
    - Resize
    - CenterCrop
    - Normalize
    """
    
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                input_size, 
                scale=(0.9, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
```

---

## 推理优化

### 量化

#### 4bit 量化

```python
# BitsAndBytes 配置

from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM-o-2_6-int4",
    quantization_config=quant_config,
    device_map="auto"
)
```

**效果**:
- 模型大小：18GB → 5GB
- 速度：提升 1.5x
- 精度损失：<1%

### vLLM 加速

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="openbmb/MiniCPM-o-2_6",
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
    max_model_len=2048
)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.9,
    max_tokens=512
)

outputs = llm.generate(prompts, sampling_params)
```

**速度提升**:
- Batch=1: 20 tokens/s
- Batch=8: 100 tokens/s

### llama.cpp GGUF

```bash
# 量化模型
./quantize minicpm-o-f16.gguf minicpm-o-q4_k_m.gguf Q4_K_M

# 推理
./main -m minicpm-o-q4_k_m.gguf \
  -p "<image>\n描述这张图片" \
  --image test.jpg \
  -t 8 \
  -ngl 32
```

**性能** (M1 Max):
- Q4_K_M: 15 tokens/s
- Q5_K_M: 12 tokens/s
- Q8_0: 8 tokens/s

---

## 附录：关键文件

| 文件 | 作用 |
|------|------|
| `omnilmm/model/omnilmm.py` | 主模型 |
| `omnilmm/model/resampler.py` | Resampler |
| `omnilmm/model/utils.py` | 工具函数 |
| `omnilmm/train/train_utils.py` | 训练工具 |
| `chat.py` | 推理入口 |

---

_MiniCPM-o 模型架构文档结束_
