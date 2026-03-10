# MiniCPM-o 训练数据与评测方法

> 📖 文档生成者：小阅 (Xiaoyue)  
> 📅 生成时间：2026-03-10

---

## 训练数据构造方法

### 预训练数据

#### 1. 图像 - 文本对

**数据来源**:
- LLaVA-665K (665K 样本)
- COCO Captions (120K)
- Flickr30k (31K)
- Visual Genome (108K)

**数据格式**:
```json
[
  {
    "id": "000001",
    "image": "coco/train2014/000001.jpg",
    "conversations": [
      {
        "role": "user",
        "content": "<image>\n描述这张图片"
      },
      {
        "role": "assistant",
        "content": "图片中展示了一个..."
      }
    ]
  }
]
```

#### 2. VQA 数据

**来源**:
- VQAv2 (11K 训练)
- GQA (1.7M)
- OK-VQA (9K)

**格式**:
```json
{
  "id": "vqav2_001",
  "image": "path/to/image.jpg",
  "question": "图片中有多少个人？",
  "answer": "有 3 个人"
}
```

### 指令微调数据

#### 构造方法

```python
# 指令数据构造脚本

import json
from PIL import Image

class InstructionDataBuilder:
    def __init__(self):
        self.templates = {
            "description": [
                "描述这张图片",
                "这张图片展示了什么？",
                "请详细说明图片内容",
            ],
            "ocr": [
                "图片中的文字是什么？",
                "识别图中的文本",
                "提取图片中的所有文字",
            ],
            "reasoning": [
                "为什么这个人看起来很开心？",
                "推测这张图片的拍摄场景",
                "分析图片中人物的关系",
            ],
            "counting": [
                "图片中有几个物体？",
                "数一数有多少个...",
                "图中有多少个人/车/动物？",
            ]
        }
    
    def build_data(self, image_path, annotations):
        """构建指令数据"""
        samples = []
        
        for ann in annotations:
            # 随机选择指令模板
            task_type = ann.get("type", "description")
            question = random.choice(self.templates[task_type])
            
            # 构建对话
            conversation = [
                {
                    "role": "user",
                    "content": f"<image>\n{question}"
                },
                {
                    "role": "assistant",
                    "content": ann["answer"]
                }
            ]
            
            sample = {
                "id": ann["id"],
                "image": image_path,
                "conversations": conversation
            }
            
            samples.append(sample)
        
        return samples
    
    def save(self, samples, output_file):
        """保存数据"""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

# 使用示例
builder = InstructionDataBuilder()
samples = builder.build_data("image.jpg", annotations)
builder.save(samples, "instruction_data.json")
```

### 数据增强

```python
# 数据增强方法

class DataAugmentation:
    def __init__(self):
        pass
    
    def paraphrase_question(self, question):
        """问题改写"""
        # 同义词替换
        synonyms = {
            "描述": ["说明", "介绍", "描绘"],
            "什么": ["哪些内容", "什么东西"],
            "多少": ["几个", "多少个"],
        }
        
        for word, syns in synonyms.items():
            if word in question:
                question = question.replace(
                    word, 
                    random.choice(syns)
                )
        
        return question
    
    def add_context(self, question, context):
        """添加上下文"""
        return f"{context}\n\n{question}"
    
    def multi_turn(self, conversations):
        """多轮对话扩展"""
        # 基于单轮对话生成多轮
        multi_turn = []
        
        # 添加后续问题
        follow_ups = [
            "为什么？",
            "能详细说说吗？",
            "还有其他什么？",
            "这说明了什么？",
        ]
        
        for conv in conversations:
            if len(conv) == 2:  # 单轮
                # 添加 1-2 轮后续问题
                num_follow = random.randint(1, 2)
                for i in range(num_follow):
                    follow_up = random.choice(follow_ups)
                    conv.append({
                        "role": "user",
                        "content": follow_up
                    })
                    conv.append({
                        "role": "assistant",
                        "content": f"基于前面的内容，{follow_up}..."
                    })
            
            multi_turn.append(conv)
        
        return multi_turn
```

---

## 评测数据构造方法

### 评测基准

#### 1. OpenCompass

**包含数据集**:
- MMBench (11 个维度)
- SEED-Bench (多模态理解)
- LLaVA Bench (开放对话)

#### 2. 专项评测

| 任务 | 数据集 | 指标 |
|------|--------|------|
| VQA | VQAv2 | Accuracy |
| OCR | OCRBench | Accuracy |
| 文档理解 | DocVQA | ANLS |
| 图表理解 | ChartQA | Accuracy |
| 数学推理 | MathVista | Accuracy |

### 测试集划分

```python
# 测试集划分脚本

from sklearn.model_selection import train_test_split

def create_test_set(data, output_dir):
    """
    创建分层测试集
    
    按任务类型分层:
    - 描述 (30%)
    - 推理 (25%)
    - OCR (20%)
    - 计数 (15%)
    - 其他 (10%)
    """
    
    # 按任务分组
    groups = {
        "description": [],
        "reasoning": [],
        "ocr": [],
        "counting": [],
        "other": []
    }
    
    for sample in data:
        task_type = sample.get("type", "other")
        groups[task_type].append(sample)
    
    # 每组抽取
    test_set = []
    for task, samples in groups.items():
        n_test = max(50, int(len(samples) * 0.2))
        test_samples = random.sample(samples, n_test)
        
        for s in test_samples:
            s["task"] = task
            test_set.append(s)
    
    # 保存
    with open(f"{output_dir}/test.jsonl", "w") as f:
        for s in test_set:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    return test_set
```

---

## 评测方法

### 客观指标

```python
# 评测指标计算

class EvaluationMetrics:
    def __init__(self):
        pass
    
    def calculate_accuracy(self, predictions, references):
        """
        计算准确率
        
        Args:
            predictions: 预测答案列表
            references: 真实答案列表
        
        Returns:
            accuracy: 准确率
        """
        correct = 0
        
        for pred, ref in zip(predictions, references):
            # 归一化
            pred_norm = self.normalize_answer(pred)
            ref_norm = self.normalize_answer(ref)
            
            # 匹配
            if pred_norm == ref_norm:
                correct += 1
            elif ref_norm in pred_norm:
                correct += 0.5  # 部分匹配
        
        accuracy = correct / len(predictions) * 100
        return accuracy
    
    def normalize_answer(self, text):
        """答案归一化"""
        import re
        
        # 转小写
        text = text.lower()
        
        # 移除标点
        text = re.sub(r'[^\w\s]', '', text)
        
        # 移除多余空格
        text = ' '.join(text.split())
        
        return text
    
    def calculate_bleu(self, predictions, references):
        """BLEU 分数"""
        from nltk.translate.bleu_score import sentence_bleu
        
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]
            
            bleu = sentence_bleu(ref_tokens, pred_tokens)
            bleu_scores.append(bleu)
        
        return sum(bleu_scores) / len(bleu_scores) * 100
    
    def calculate_anls(self, predictions, references):
        """
        ANLS (Average Normalized Levenshtein Similarity)
        
        用于 OCR 和文档理解任务
        """
        from difflib import SequenceMatcher
        
        anls_scores = []
        
        for pred, ref in zip(predictions, references):
            # 归一化
            pred_norm = self.normalize_answer(pred)
            ref_norm = self.normalize_answer(ref)
            
            # 相似度
            similarity = SequenceMatcher(
                None, 
                pred_norm, 
                ref_norm
            ).ratio()
            
            # ANLS (阈值 0.5)
            if similarity < 0.5:
                anls = 0.0
            else:
                anls = similarity
            
            anls_scores.append(anls)
        
        return sum(anls_scores) / len(anls_scores) * 100
```

### 分场景评测

```python
# 分场景评测

class ScenarioEvaluator:
    def __init__(self, model):
        self.model = model
    
    def evaluate_by_scenario(self, test_data):
        """分场景评测"""
        
        scenarios = {
            "single_object": [],      # 单物体
            "multi_object": [],       # 多物体
            "scene_text": [],         # 场景文字
            "document": [],           # 文档
            "chart": [],              # 图表
            "math": [],               # 数学
        }
        
        # 分组
        for sample in test_data:
            scenario = sample.get("scenario", "single_object")
            scenarios[scenario].append(sample)
        
        # 评测
        results = {}
        for scenario, samples in scenarios.items():
            predictions = []
            references = []
            
            for sample in samples:
                # 推理
                pred = self.model.generate(
                    sample["image"],
                    sample["question"]
                )
                
                predictions.append(pred)
                references.append(sample["answer"])
            
            # 计算指标
            acc = self.calculate_accuracy(predictions, references)
            
            results[scenario] = {
                "accuracy": acc,
                "count": len(samples)
            }
        
        return results
```

### 人工评测 (MOS)

```python
# 人工评测界面

import gradio as gr

class ManualEvaluation:
    def __init__(self):
        self.ratings = []
    
    def create_interface(self):
        """创建评测界面"""
        with gr.Blocks() as demo:
            gr.Markdown("# MiniCPM-o 人工评测")
            
            with gr.Row():
                image = gr.Image(label="测试图像")
                question = gr.Textbox(label="问题")
            
            with gr.Row():
                pred_answer = gr.Textbox(label="模型回答")
                ref_answer = gr.Textbox(label="参考答案")
            
            # 评分
            with gr.Row():
                accuracy_score = gr.Slider(
                    1, 5, step=1,
                    label="准确性 (1=完全错误，5=完全正确)"
                )
                fluency_score = gr.Slider(
                    1, 5, step=1,
                    label="流畅性 (1=不流畅，5=非常流畅)"
                )
                relevance_score = gr.Slider(
                    1, 5, step=1,
                    label="相关性 (1=不相关，5=非常相关)"
                )
            
            comments = gr.Textbox(
                label="备注",
                placeholder="请说明评分理由..."
            )
            
            submit_btn = gr.Button("提交", variant="primary")
        
        return demo
    
    def save_rating(self, scores, comments):
        """保存评分"""
        self.ratings.append({
            "accuracy": scores[0],
            "fluency": scores[1],
            "relevance": scores[2],
            "comments": comments
        })
        
        # 保存到文件
        with open("mos_ratings.jsonl", "a") as f:
            f.write(json.dumps(self.ratings[-1]) + "\n")
```

---

_训练数据与评测方法文档结束_
