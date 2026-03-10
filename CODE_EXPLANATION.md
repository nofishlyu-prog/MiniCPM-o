# MiniCPM-o 代码逻辑详解

> 📖 文档生成者：小阅 (Xiaoyue)  
> 📅 生成时间：2026-03-10  
> 🎯 项目：MiniCPM-o 4.5

---

## 📑 目录

1. [项目概览](#项目概览)
2. [核心架构](#核心架构)
3. [模型代码详解](#模型代码详解)
4. [推理流程](#推理流程)
5. [微调代码](#微调代码)
6. [评测代码](#评测代码)
7. [Web Demo](#web-demo)
8. [部署代码](#部署代码)

---

## 项目概览

### MiniCPM-o 是什么？

MiniCPM-o 是 OpenBMB 推出的**端侧多模态大语言模型系列**，最新模型 MiniCPM-o 4.5 的特点：

- **9B 参数** - 端侧高效部署
- **视觉 + 语音 + 全双工** - 多模态实时交互
- **Gemini 2.5 Flash 级别** - 开源社区最强之一
- **支持 30+ 语言** - 多语言能力强

### 项目结构

```
MiniCPM-o/
├── chat.py                          # 推理入口
├── omnilmm/                         # 核心模型代码
│   ├── model/
│   │   ├── omnilmm.py               # 主模型定义 ⭐
│   │   ├── resampler.py             # 视觉重采样
│   │   └── utils.py                 # 工具函数
│   ├── train/                       # 训练代码
│   └── conversation.py              # 对话管理
│
├── finetune/                        # 微调脚本
│   ├── finetune.py                  # 全量微调
│   ├── finetune_lora.py             # LoRA 微调
│   └── README.md                    # 微调指南
│
├── eval_mm/                         # 多模态评测
│   ├── vqaeval/                     # VQA 评测
│   └── vlmevalkit/                  # VLMEval 集成
│
├── web_demos/                       # Web 演示
│   ├── minicpm-o_2.6/
│   │   ├── model_server.py          # 模型服务
│   │   └── chatbot_web_demo_o2.6.py # Web 聊天界面
│   └── ...
│
├── assets/                          # 资源文件
├── docs/                            # 文档
├── requirements.txt                 # 依赖
└── README.md                        # 主文档
```

### 技术栈

```
深度学习框架：PyTorch 2.0+
Transformer 库：Transformers 4.37+
视觉编码器：EVA-02 (timm)
语言模型：Mistral / Qwen2.5
推理加速：vLLM, llama.cpp, Ollama
微调框架：DeepSpeed, LoRA
```

---

## 核心架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      多模态输入                              │
│         图像 / 视频 / 文本 / 语音                            │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
         ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Vision     │ │   Audio     │ │    Text     │
│  Encoder    │ │   Encoder   │ │  Embedding  │
│  (EVA-02)   │ │  (Whisper)  │ │             │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       │ 视觉特征       │ 音频特征       │ 文本嵌入
       │ (14x14)       │ (Whisper)     │
       │               │               │
       ▼               ▼               │
┌──────────────────────────────┐       │
│      Resampler + Projector   │       │
│      (特征对齐 + 投影)          │       │
└──────────────┬───────────────┘       │
               │                       │
               │ 多模态嵌入             │
               ▼                       │
        ┌──────────────────┐          │
        │  LLM Backbone    │◄─────────┘
        │  (Qwen2.5-8B)    │
        │                  │
        │  - Self-Attention│
        │  - Cross-Attention
        │  - FFN           │
        └────────┬─────────┘
                 │
                 │ 隐藏状态
                 ▼
        ┌──────────────────┐
        │   Output Head    │
        │  - Text (Logits) │
        │  - Speech (TTS)  │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │    输 出          │
        │  文本 / 语音       │
        └──────────────────┘
```

### 核心组件

| 组件 | 文件 | 作用 |
|------|------|------|
| Vision Encoder | `omnilmm/model/omnilmm.py` | EVA-02 视觉编码 |
| Resampler | `omnilmm/model/resampler.py` | 特征重采样对齐 |
| LLM Backbone | Transformers | Qwen2.5 / Mistral |
| Projector | `omnilmm/model/omnilmm.py` | 模态投影 |
| Conversation | `omnilmm/conversation.py` | 对话管理 |

---

## 模型代码详解

### 1. 主模型类 (OmniLMMModel)

**文件**: `omnilmm/model/omnilmm.py`

```python
class OmniLMMModel(MistralModel):
    """
    MiniCPM-o 核心模型类
    
    继承自 MistralModel，添加视觉模块
    """
    
    config_class = OmniLMMConfig
    
    def __init__(self, config: OmniLMMConfig, 
                 mm_vision_tower=None, 
                 mm_hidden_size=None, 
                 tune_clip=True):
        super(OmniLMMModel, self).__init__(config)
        
        if hasattr(config, "mm_vision_tower"):
            # 创建视觉模块
            vision_tower, resampler = create_vision_module(config)
            
            # HACK: for FSDP (分布式训练)
            self.vision_tower = [vision_tower]
            self.resampler = resampler
            
            if tune_clip:
                self.vision_tower = self.vision_tower[0]
    
    def initialize_vision_modules(self, vision_tower, 
                                   no_randaug, 
                                   num_query, 
                                   image_size, 
                                   tune_clip=False):
        """
        初始化视觉模块
        
        Args:
            vision_tower: 视觉编码器名称
            no_randaug: 是否使用数据增强
            num_query: 查询 token 数量 (64)
            image_size: 输入图像尺寸
            tune_clip: 是否微调视觉编码器
        """
        self.config.mm_vision_tower = vision_tower
        self.config.use_mm_proj = True
        self.config.num_query = num_query
        self.config.image_size = image_size
        
        # 加载预训练权重
        if not hasattr(self, 'vision_tower'):
            vision_tower, resampler = create_vision_module(self.config)
            
            # 加载 EVA-02 预训练权重
            state_dict = torch.load(
                '/tt/data/public/multimodal/.../eva02_enormous_patch14_clip_224.laion2b_plus.pt'
            )
            vision_tower.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
        
        # 构建图像变换
        train_img_transform = build_transform(
            is_train=True, 
            randaug=not no_randaug, 
            input_size=self.config.image_size, 
            std_mode='OPENAI_CLIP'
        )
        eval_img_transform = build_transform(
            is_train=False, 
            input_size=self.config.image_size, 
            std_mode='OPENAI_CLIP'
        )
        
        return dict(
            image_processor=(train_img_transform, eval_img_transform),
            image_token_len=num_query,
            vision_config=self.vision_config
        )
```

### 2. 视觉模块创建

```python
def create_vision_module(config):
    """
    创建视觉编码模块
    
    使用 EVA-02 Enormous (Vision Transformer)
    """
    # 创建 EVA-02 模型
    vision_tower = timm.create_model(
        'eva02_enormous_patch14_clip_224.laion2b_plus',
        pretrained=False,
        num_classes=0,              # 无分类头
        dynamic_img_size=True,      # 动态尺寸支持
        dynamic_img_pad=True,
    )
    
    # 使用倒数第二层输出
    vision_tower.blocks[-1] = Identity()
    
    # 创建 Resampler (特征重采样)
    embed_dim = config.hidden_size
    resampler = Resampler(
        grid_size=int(math.sqrt(config.num_query)),  # 8x8 = 64
        embed_dim=embed_dim,
        num_heads=embed_dim // 128,
        kv_dim=vision_tower.embed_dim,
    )
    
    return vision_tower, resampler


class Identity(torch.nn.Identity):
    """恒等映射层 (用于替换最后一层)"""
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return super().forward(input)
```

### 3. 视觉特征提取

```python
def get_vision_embedding(self, pixel_values):
    """
    提取视觉特征
    
    Args:
        pixel_values: 图像张量 (B, C, H, W)
    
    Returns:
        res: 重采样后的视觉特征 (B, num_query, embed_dim)
    """
    # 获取视觉编码器
    if isinstance(self.vision_tower, list):
        vision_tower = self.vision_tower[0]  # HACK: for FSDP
    else:
        vision_tower = self.vision_tower
    
    # 类型转换 (匹配位置编码 dtype)
    dtype = vision_tower.pos_embed.data.dtype
    vision_embedding = vision_tower.forward_features(
        pixel_values.type(dtype)
    )
    
    # 移除 prefix tokens (如果有)
    if hasattr(vision_tower, 'num_prefix_tokens') and \
       vision_tower.num_prefix_tokens > 0:
        vision_embedding = vision_embedding[:, 
                                            vision_tower.num_prefix_tokens:]
    
    # Resampler: 将 N 个视觉 token 压缩为 num_query 个
    res = self.resampler(vision_embedding)
    
    return res
```

### 4. 多模态嵌入融合

```python
def get_vllm_embedding(self, data):
    """
    获取 vLLM 推理用的嵌入表示
    
    Args:
        data: 包含 input_ids, pixel_values 等的字典
    
    Returns:
        inputs_embeds: 融合后的嵌入 (B, L, dim)
        vision_hidden_states: 视觉隐藏状态列表
    """
    # 1. 处理视觉特征
    if 'vision_hidden_states' not in data:
        pixel_values_list = data['pixel_values']
        vision_hidden_states = []
        
        for pixel_values in pixel_values_list:
            if len(pixel_values) > 0:
                vision_hidden_states.append(
                    self.get_vision_embedding(
                        pixel_values.unsqueeze(0)
                    )[0]
                )
            else:
                vision_hidden_states.append([])
    else:
        vision_hidden_states = data['vision_hidden_states']
    
    # 2. 获取文本嵌入
    inputs_embeds = self.embed_tokens(data['input_ids'])
    
    # 3. 视觉特征类型转换
    vision_hidden_states = [
        i.type(inputs_embeds.dtype) 
        if isinstance(i, torch.Tensor) 
        else i 
        for i in vision_hidden_states
    ]
    
    # 4. 融合视觉和文本嵌入
    new_input_embeds = []
    cur_image_idx = 0
    
    for cur_input_ids, cur_input_embeds in zip(
        data['input_ids'], 
        inputs_embeds
    ):
        # 检查是否有图像 token
        if (cur_input_ids == self.vision_config.im_patch_token).sum() == 0:
            # 非多模态样本
            cur_input_embeds = cur_input_embeds + (
                0. * dummy_image_features
            ).sum()
            new_input_embeds.append(cur_input_embeds)
            continue
        
        # 有图像，进行融合
        if self.vision_config.use_im_start_end:
            cur_image_features = vision_hidden_states[cur_image_idx]
            num_patches = cur_image_features.shape[0]
            
            # 找到图像起始 token
            image_start_tokens = torch.where(
                cur_input_ids == self.vision_config.im_start_token
            )[0]
            
            for image_start_token_pos in image_start_tokens:
                # 提取图像特征
                cur_image_features = vision_hidden_states[
                    cur_image_idx
                ].to(device=cur_input_embeds.device)
                
                num_patches = cur_image_features.shape[0]
                
                # 验证图像结束 token
                if cur_input_ids[
                    image_start_token_pos + num_patches + 1
                ] != self.vision_config.im_end_token:
                    raise ValueError(
                        "The image end token should follow the image start token."
                    )
                
                # 拼接嵌入：[文本前缀] + [图像特征] + [文本后缀]
                cur_new_input_embeds = torch.cat(
                    (
                        cur_input_embeds[:image_start_token_pos+1],
                        cur_image_features,
                        cur_input_embeds[
                            image_start_token_pos + num_patches + 1:
                        ]
                    ), 
                    dim=0
                )
                
                cur_image_idx += 1
            
            new_input_embeds.append(cur_new_input_embeds)
        else:
            raise NotImplementedError
    
    # 堆叠为 batch
    inputs_embeds = torch.stack(new_input_embeds, dim=0)
    
    return inputs_embeds, vision_hidden_states
```

### 5. 前向传播

```python
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    images: Optional[torch.FloatTensor] = None,
    return_dict: Optional[bool] = None,
    **kwargs
) -> Union[Tuple, BaseModelOutputWithPast]:
    """
    模型前向传播
    
    支持纯文本和多模态输入
    """
    # 使用默认参数
    output_attentions = (
        output_attentions 
        if output_attentions is not None 
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states 
        if output_hidden_states is not None 
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    
    return_dict = (
        return_dict 
        if return_dict is not None 
        else self.config.use_return_dict
    )
    
    # 如果有图像，获取视觉嵌入
    if images is not None:
        vision_outputs = self.get_vision_embedding(images)
    
    # 如果有 input_ids 但没有 inputs_embeds，且有多模态 token
    if input_ids is not None and inputs_embeds is None:
        # 检查是否有图像
        if (input_ids == self.vision_config.im_patch_token).sum() > 0:
            # 需要准备 pixel_values
            pixel_values = kwargs.get('pixel_values', None)
            
            if pixel_values is not None:
                # 获取融合嵌入
                data = {
                    'input_ids': input_ids,
                    'pixel_values': pixel_values
                }
                inputs_embeds, _ = self.get_vllm_embedding(data)
    
    # 调用父类 MistralModel 的 forward
    outputs = super().forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    
    return outputs
```

---

## 推理流程

### chat.py 推理入口

```python
# chat.py 核心代码

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def init_omni_lmm(model_path):
    """
    初始化 OmniLMM 模型
    
    Args:
        model_path: 模型路径
    
    Returns:
        model, image_processor, image_token_len, tokenizer
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    disable_torch_init()  # 禁用 torch 初始化加速
    
    model_name = os.path.expanduser(model_path)
    print(f'Load omni_lmm model and tokenizer from {model_name}')
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        model_max_length=2048
    )
    
    # 加载模型
    model = OmniLMMForCausalLM.from_pretrained(
        model_name, 
        tune_clip=True, 
        torch_dtype=torch.bfloat16
    ).to(device='cuda', dtype=torch.bfloat16)
    
    # 构建图像处理器
    image_processor = build_transform(
        is_train=False, 
        input_size=model.model.config.image_size, 
        std_mode='OPENAI_CLIP'
    )
    
    # 添加特殊 token
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    assert mm_use_im_start_end
    
    tokenizer.add_tokens(
        [
            DEFAULT_IMAGE_PATCH_TOKEN, 
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN
        ], 
        special_tokens=True
    )
    
    # 配置视觉模块
    vision_config = model.model.vision_config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    vision_config.im_start_token, vision_config.im_end_token = (
        tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )
    )
    image_token_len = model.model.config.num_query
    
    return model, image_processor, image_token_len, tokenizer


def expand_question_into_multimodal(
    question_text, 
    image_token_len, 
    im_st_token, 
    im_ed_token, 
    im_patch_token
):
    """
    将问题扩展为多模态格式
    
    在文本中插入图像 token
    """
    if '<image>' in question_text[0]['content']:
        # 替换<image>为实际的图像 token 序列
        question_text[0]['content'] = question_text[0]['content'].replace(
            '<image>', 
            im_st_token + im_patch_token * image_token_len + im_ed_token
        )
    else:
        # 在开头添加图像 token
        question_text[0]['content'] = (
            im_st_token + 
            im_patch_token * image_token_len + 
            im_ed_token + 
            '\n' + 
            question_text[0]['content']
        )
    
    return question_text


def wrap_question_for_omni_lmm(question, image_token_len, tokenizer):
    """
    包装问题为 OmniLMM 格式
    
    Args:
        question: 对话列表 [{"role": "user", "content": "..."}]
        image_token_len: 图像 token 长度 (64)
        tokenizer: 分词器
    
    Returns:
        data_dict: {"input_ids": ..., "labels": ...}
    """
    # 扩展为多模态格式
    question = expand_question_into_multimodal(
        question, 
        image_token_len, 
        DEFAULT_IM_START_TOKEN, 
        DEFAULT_IM_END_TOKEN, 
        DEFAULT_IMAGE_PATCH_TOKEN
    )
    
    # 预处理
    conversation = question
    data_dict = omni_preprocess(
        sources=[conversation],
        tokenizer=tokenizer,
        generation=True
    )
    
    data_dict = dict(
        input_ids=data_dict["input_ids"][0],
        labels=data_dict["labels"][0]
    )
    
    return data_dict


class OmniLMM12B:
    """OmniLMM 推理封装类"""
    
    def __init__(self, model_path) -> None:
        model, img_processor, image_token_len, tokenizer = (
            init_omni_lmm(model_path)
        )
        
        self.model = model
        self.image_token_len = image_token_len
        self.image_transform = img_processor
        self.tokenizer = tokenizer
        self.model.eval()
    
    def decode(self, image, input_ids):
        """
        推理生成
        
        Args:
            image: 图像张量 (C, H, W)
            input_ids: 输入 ID (L,)
        
        Returns:
            output_text: 生成的文本
        """
        with torch.inference_mode():
            output = self.model.generate_vllm(
                input_ids=input_ids.unsqueeze(0).cuda(),
                images=image.unsqueeze(0).half().cuda(),
                temperature=0.6,
                max_new_tokens=512,
                # num_beams=1,
                do_sample=True,
                top_p=0.9,
            )
            
            output_text = self.tokenizer.decode(
                output[0], 
                skip_special_tokens=True
            )
            
            return output_text
    
    def chat(self, image_path, question):
        """
        多模态对话
        
        Args:
            image_path: 图像路径
            question: 问题文本
        
        Returns:
            response: 模型回复
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 图像预处理
        image_tensor = self.image_transform(image)
        
        # 包装问题
        question_wrap = wrap_question_for_omni_lmm(
            [{"role": "user", "content": question}],
            self.image_token_len,
            self.tokenizer
        )
        
        input_ids = question_wrap['input_ids'].unsqueeze(0).cuda()
        
        # 推理
        response = self.decode(image_tensor, input_ids)
        
        return response
```

### 完整推理流程

```
1. 模型加载
   │
   ├─→ 加载 tokenizer
   ├─→ 加载 OmniLMMForCausalLM
   ├─→ 加载 EVA-02 视觉权重
   ├─→ 添加特殊 token
   └─→ 移动到 GPU (bfloat16)
   
2. 图像预处理
   │
   ├─→ 读取图像 (PIL)
   ├─→ Resize + Crop
   ├─→ 归一化 (OPENAI_CLIP std)
   └─→ 转换为 Tensor (C, H, W)
   
3. 文本处理
   │
   ├─→ 构建对话 [{"role": "user", "content": "<image>\n..."}]
   ├─→ 替换<image>为 token 序列
   │   <im_start> + <im_patch>*64 + <im_end>
   ├─→ omni_preprocess 处理
   └─→ 获取 input_ids, labels
   
4. 视觉特征提取
   │
   ├─→ EVA-02 编码
   │   输入：(1, 3, 224, 224)
   │   输出：(N, 1792) N≈256
   ├─→ Resampler 重采样
   │   输入：(N, 1792)
   │   输出：(64, 4096)
   └─→ 投影到 LLM 维度
   
5. 多模态融合
   │
   ├─→ 文本嵌入 (L, 4096)
   ├─→ 找到<im_start>位置
   ├─→ 插入视觉特征 (64, 4096)
   └─→ 拼接：[文本前缀] + [视觉] + [文本后缀]
   
6. LLM 推理
   │
   ├─→ Mistral/Qwen 前向
   ├─→ Self-Attention
   ├─→ Cross-Attention (视觉)
   ├─→ FFN
   └─→ 输出 logits
   
7. 解码生成
   │
   ├─→ Sampling (top_p=0.9, temp=0.6)
   ├─→ 生成 token 序列
   └─→ tokenizer 解码为文本
```

---

## 微调代码

### 全量微调

**文件**: `finetune/finetune.py`

```python
# 全量微调脚本核心

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import json

from omnilmm.model.omnilmm import OmniLMMForCausalLM, OmniLMMConfig
from omnilmm.train.train_utils import (
    omni_preprocess, 
    make_supervised_data_module
)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="openbmb/MiniCPM-o-2_6"
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, 
        metadata={"help": "Path to the training data."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length."
        },
    )
    tune_clip: bool = field(default=True)

def train():
    """主训练函数"""
    # 解析参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 加载模型
    model = OmniLMMForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        tune_clip=training_args.tune_clip,
        torch_dtype=torch.bfloat16,
    )
    
    # 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
    )
    
    # 添加特殊 token
    tokenizer.add_tokens(
        ['<im_patch>', '<im_start>', '<im_end>'], 
        special_tokens=True
    )
    
    # 准备数据
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, 
        data_args=data_args
    )
    
    # 初始化 Trainer
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
```

### LoRA 微调

**文件**: `finetune/finetune_lora.py`

```python
# LoRA 微调核心

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def train_lora():
    """LoRA 微调"""
    # 加载基础模型
    model = OmniLMMForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_4bit=True,  # 4bit 量化
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # 准备 K-bit 训练
    model = prepare_model_for_kbit_training(model)
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r=16,  # LoRA 秩
        lora_alpha=32,
        target_modules=[
            "q_proj", "v_proj",  # Attention 层
            "gate_proj", "up_proj", "down_proj"  # FFN 层
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 输出：trainable params: 8.2M || all params: 9.0B || trainable%: 0.09%
    
    # 训练...
    trainer.train()
    
    # 保存 LoRA 权重
    model.save_pretrained(training_args.output_dir)
```

### 数据预处理

```python
# omnilmm/train/train_utils.py

def omni_preprocess(sources, tokenizer, generation=False):
    """
    OmniLMM 数据预处理
    
    Args:
        sources: 对话列表
        tokenizer: 分词器
        generation: 是否用于生成
    
    Returns:
        dict: {"input_ids": ..., "labels": ...}
    """
    # 系统提示
    system_prompt = "你是一个有用的 AI 助手。"
    
    # 角色映射
    role_map = {"user": "Human", "assistant": "Assistant"}
    
    input_ids = []
    labels = []
    
    for conversation in sources:
        # 构建对话文本
        conversation_text = ""
        
        for i, turn in enumerate(conversation):
            role = role_map[turn["role"]]
            content = turn["content"]
            
            if i == 0:
                # 第一轮：添加系统提示
                conversation_text += (
                    f"{system_prompt}\n\n"
                    f"{role}: {content}\n\n"
                )
            else:
                conversation_text += f"{role}: {content}\n\n"
        
        # 分词
        ids = tokenizer.encode(
            conversation_text,
            add_special_tokens=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        )
        
        input_ids.append(ids)
        
        # 构建 labels (只监督 assistant 部分)
        label = [-100] * len(ids)  # -100 表示忽略
        
        # 找到 assistant 回复的位置
        for turn in conversation:
            if turn["role"] == "assistant":
                # 标记 assistant 回复的 token
                assistant_text = f"Assistant: {turn['content']}"
                assistant_ids = tokenizer.encode(
                    assistant_text,
                    add_special_tokens=False
                )
                
                # 在 input_ids 中找到对应位置
                start_idx = find_sublist(ids, assistant_ids)
                if start_idx != -1:
                    end_idx = start_idx + len(assistant_ids)
                    label[start_idx:end_idx] = ids[start_idx:end_idx]
        
        labels.append(label)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }


def make_supervised_data_module(tokenizer, data_args):
    """
    构建监督微调数据
    
    Args:
        tokenizer: 分词器
        data_args: 数据参数
    
    Returns:
        data_module: {"train_dataset": ..., "eval_dataset": ...}
    """
    # 加载数据
    with open(data_args.data_path, 'r') as f:
        data_list = json.load(f)
    
    # 划分训练/验证
    train_data = data_list[:int(len(data_list)*0.9)]
    eval_data = data_list[int(len(data_list)*0.9):]
    
    # 预处理
    train_dataset = SupervisedDataset(
        train_data, 
        tokenizer, 
        data_args
    )
    eval_dataset = SupervisedDataset(
        eval_data, 
        tokenizer, 
        data_args
    )
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


class SupervisedDataset(torch.utils.data.Dataset):
    """监督微调数据集"""
    
    def __init__(self, data_list, tokenizer, data_args):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.data_args = data_args
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        # 获取图像
        image_path = sample["image"]
        image = Image.open(image_path).convert('RGB')
        
        # 图像预处理
        image_transform = build_transform(
            is_train=True,
            input_size=224,
            std_mode='OPENAI_CLIP'
        )
        image_tensor = image_transform(image)
        
        # 文本预处理
        conversation = sample["conversations"]
        data_dict = omni_preprocess(
            [conversation], 
            self.tokenizer
        )
        
        return {
            "input_ids": data_dict["input_ids"][0],
            "labels": data_dict["labels"][0],
            "pixel_values": image_tensor,
        }
```

---

## 评测代码

### VQA 评测

**文件**: `eval_mm/vqaeval/models/MiniCPM/minicpmv.py`

```python
# VQA 评测模型封装

class MiniCPMV:
    """MiniCPM-V/O 评测封装"""
    
    def __init__(self, model_path):
        # 加载模型
        self.model, self.processor = self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载模型和处理器"""
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda"
        )
        
        return model, processor
    
    def generate(self, image, question):
        """
        生成答案
        
        Args:
            image: PIL 图像
            question: 问题文本
        
        Returns:
            answer: 生成的答案
        """
        # 构建消息
        messages = [{
            'role': 'user',
            'content': f'<image>\n{question}'
        }]
        
        # 准备输入
        prompt = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            images=[image],
            text=prompt,
            return_tensors='pt'
        ).to('cuda')
        
        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        
        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(
                inputs.input_ids, 
                generated_ids
            )
        ]
        
        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return answer
    
    def evaluate_vqa(self, vqa_dataset):
        """
        VQA 评测
        
        Args:
            vqa_dataset: VQA 数据集
        
        Returns:
            accuracy: 准确率
        """
        correct = 0
        total = 0
        
        for sample in vqa_dataset:
            image = sample["image"]
            question = sample["question"]
            answer = sample["answer"]
            
            # 生成预测
            pred = self.generate(image, question)
            
            # 计算准确率 (简单匹配)
            if self.match_answer(pred, answer):
                correct += 1
            
            total += 1
        
        accuracy = correct / total * 100
        return accuracy
    
    def match_answer(self, pred, answer):
        """答案匹配 (简单版本)"""
        # 归一化
        pred_norm = pred.lower().strip()
        answer_norm = answer.lower().strip()
        
        # 精确匹配
        if pred_norm == answer_norm:
            return True
        
        # 包含匹配
        if answer_norm in pred_norm:
            return True
        
        return False
```

---

## Web Demo

### 模型服务

**文件**: `web_demos/minicpm-o_2.6/model_server.py`

```python
# 模型服务 FastAPI

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

# 全局模型
model = None

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global model
    model = OmniLMM12B("openbmb/MiniCPM-o-2_6")

@app.post("/chat")
async def chat(
    file: UploadFile = File(...),
    question: str = None
):
    """
    多模态对话接口
    
    Args:
        file: 上传的图像
        question: 问题文本
    
    Returns:
        {"response": "..."}
    """
    # 读取图像
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # 推理
    response = model.chat(image, question)
    
    return {"response": response}

# 启动：uvicorn model_server:app --host 0.0.0.0 --port 8000
```

### Web 聊天界面

**文件**: `web_demos/minicpm-o_2.6/chatbot_web_demo_o2.6.py`

```python
# Gradio Web 界面

import gradio as gr

def chat_with_image(image, question, history):
    """
    图像对话
    
    Args:
        image: 上传的图像
        question: 问题
        history: 对话历史
    
    Returns:
        history: 更新后的对话历史
    """
    if image is None:
        return history + [(question, "请上传一张图像。")]
    
    if not question:
        return history + [(None, "请输入问题。")]
    
    # 调用模型
    response = model.chat(image, question)
    
    # 更新历史
    history.append((question, response))
    
    return history

# 创建界面
with gr.Blocks() as demo:
    gr.Markdown("# MiniCPM-o 多模态对话")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil", 
                label="上传图像"
            )
            question_input = gr.Textbox(
                label="问题",
                placeholder="请输入关于图像的问题..."
            )
            submit_btn = gr.Button("提交", variant="primary")
        
        with gr.Column():
            chat_output = gr.Chatbot(
                label="对话历史",
                height=400
            )
    
    # 事件绑定
    submit_btn.click(
        fn=chat_with_image,
        inputs=[image_input, question_input, chat_output],
        outputs=chat_output
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

## 部署代码

### llama.cpp 部署

**配置 Modelfile**:

```dockerfile
# Modelfile for Ollama

FROM openbmb/MiniCPM-o-2_6

# 设置参数
PARAMETER temperature 0.6
PARAMETER top_p 0.9
PARAMETER num_ctx 2048

# 系统提示
SYSTEM """
你是一个有用的 AI 助手，能够理解和分析图像。
"""
```

**Ollama 运行**:

```bash
# 拉取模型
ollama pull openbmb/minicpm-o-2_6

# 运行
ollama run openbmb/minicpm-o-2_6

# 带图像运行
ollama run openbmb/minicpm-o-2_6 "描述这张图片" --image test.jpg
```

### vLLM 部署

```python
# vLLM 推理

from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="openbmb/MiniCPM-o-2_6",
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1
)

# 采样参数
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.9,
    max_tokens=512
)

# 推理
prompts = [
    "<image>\n这张图片里有什么？",
]

outputs = llm.generate(
    prompts,
    sampling_params,
    multi_modal_data={
        "image": Image.open("test.jpg")
    }
)

for output in outputs:
    print(output.outputs[0].text)
```

---

## 附录：关键文件索引

| 文件 | 作用 | 代码量 |
|------|------|--------|
| `omnilmm/model/omnilmm.py` | 主模型定义 | ~600 行 |
| `omnilmm/model/resampler.py` | 视觉重采样 | ~100 行 |
| `omnilmm/conversation.py` | 对话管理 | ~400 行 |
| `chat.py` | 推理入口 | ~200 行 |
| `finetune/finetune.py` | 全量微调 | ~300 行 |
| `finetune/finetune_lora.py` | LoRA 微调 | ~200 行 |
| `eval_mm/vqaeval/models/MiniCPM/minicpmv.py` | VQA 评测 | ~150 行 |
| `web_demos/minicpm-o_2.6/model_server.py` | 模型服务 | ~100 行 |

---

_MinimumCPM-o 代码逻辑详解文档结束_
