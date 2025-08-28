
import torch
import omegaconf
from datasets import Dataset
from typing import List, Tuple, Union
from torch.utils.data import DataLoader
import numpy as np
# from dataset.base import MMSafetyBenchDataset,MSSBenchDataset,ShareGPT4vDataset,SIUO
from transformers import TextStreamer, MllamaForConditionalGeneration, AutoProcessor, HfArgumentParser,AutoModelForVision2Seq,PreTrainedTokenizer,AutoModelForImageTextToText
from dataclasses import dataclass,field
from collections.abc import Sequence
from PIL import Image
from typing import Optional,Union,Literal
import os, json
# from src.utils import *
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel

SYSTEM_PROMPT : str = "You are a helpful, respectful and honest assistant."

class MMSafetyBenchDataset(Dataset):
    dataset_ids: str = "mm-safety-bench"
    dataset_categories_list: Sequence[str] = ['04-Physical_Harm', '05-EconomicHarm', '12-Health_Consultation', '10-Legal_Opinion', '06-Fraud', '13-Gov_Decision', '09-Privacy_Violence', '03-Malware_Generation', '08-Political_Lobbying', '11-Financial_Advice', '07-Sex', '02-HateSpeech', '01-Illegal_Activitiy']

    name_list : str = '/root/data/MM-SafetyBench/processed_questions'
    image_dir : str ='/root/data/MM-SafetyBench/images'

    def __init__(self,num_select) -> None:
        self.annotations = []
        self.num_selet = num_select

    def get_dataset(self):
        for i in self.dataset_categories_list:
            self.dataset_categories = i
            self.image_dir_ = os.path.join(self.image_dir, self.dataset_categories, "SD")
            self.name_list_ = os.path.join(self.name_list, f"{self.dataset_categories}.json")
            data = json.load(open(self.name_list_, "r"))

            for idx, entry in data.items():
                self.annotations.append(
                    {
                        "question": SYSTEM_PROMPT + entry["Rephrased Question(SD)"],
                        "image_path": os.path.join(self.image_dir_, f"{idx}.jpg"),
                    }
                )
        self.dataset = Dataset.from_list(self.annotations)
        self.dataset = self.dataset.shuffle(seed=42).select(range(self.num_selet))
        print("mmsafetybenh:",len(self.dataset))
        return self.dataset

class SharedGPT4vDataset(Dataset):
    name_list : str = '/fs-computility/ai-shen/kilab-shared/benchmarks/MMMU_val/benchmark.jsonl'
    image_dir : str ='/fs-computility/ai-shen/kilab-shared/benchmarks/MMMU_val'

    def __init__(self,num_select) -> None:
        self.annotations = []
        self.num_selet = num_select

    def get_dataset(self):
        with open(self.name_list, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                image_path = os.path.join(self.image_dir, data["index"] )
                self.annotations.append(
                    {
                        "question": SYSTEM_PROMPT + data["question"],
                        "image_path": image_path,
                    }
                )
        self.dataset = Dataset.from_list(self.annotations)
        self.dataset = self.dataset.shuffle(seed=42).select(range(self.num_selet))
        print("mmmu_val:",len(self.dataset))

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
batch_size = 4  # tune for your GPU
# model_name = "/fs-computility/ai-shen/shared/VLM/QwenVL/Qwen2.5-VL-7B-Instruct"
model_name = "/fs-computility/ai-shen/shared/hf-hub/Qwen2.5-VL-7B-Instruct"



# ----------------------------
# 1) Load model & processor
# ----------------------------
processor = AutoProcessor.from_pretrained(model_name,trust_remote_code=True)
processor.tokenizer = AutoTokenizer.from_pretrained(model_name)
processor.tokenizer.padding_side = "left"
model = AutoModelForVision2Seq.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")

# ----------------------------
@torch.no_grad()
def embed_dataset(
    ds: Dataset,
    text_key: str = "question",
    image_key: str = "image_path",
    batch_size: int = 4
) -> np.ndarray:
    """
    Returns a (N, D) array: one fused embedding per (image, question) item.
    Uses last hidden states with mean-pooling over tokens to get a single vector.
    """
    embs: List[np.ndarray] = []
    N = len(ds)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = ds.select(range(start, end))

        # --- 构造 multimodal message ---
        messages = [[
            {"role": "user",
             "content": [
                 {"type": "image", "text": None, "index": 0},
                 {"type": "text", "text": ex[text_key], "index": None}
             ]
            }
        ] for ex in batch]

        # --- 文本处理 ---
        texts = [processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                 ) for message in messages]

        # --- 图片处理 ---
        images = [[Image.open(ex[image_key]).convert("RGB")] for ex in batch]

        # --- 输入编码 ---
        inputs = processor(
            text=texts,
            images=images,  # 保持原来 collator 风格
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # --- 前向计算 hidden states ---
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # --- 提取最后一层隐藏状态并 mean pooling ---
        last_hidden = outputs.hidden_states[-1]  # [B, T, D]
        pooled = last_hidden.mean(dim=1)         # [B, D]
        embs.append(pooled.float().cpu().numpy())

    return np.concatenate(embs, axis=0)

def save_embeddings(embs: np.ndarray, save_dir: str, filename: str = "embeddings.npy"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    np.save(path, embs)   # 保存为 numpy 格式
    print(f"✅ Embeddings saved to {path}")

def load_embeddings(save_dir: str, filename: str = "embeddings.npy") -> np.ndarray:
    path = os.path.join(save_dir, filename)
    embs = np.load(path)
    print(f"✅ Embeddings loaded from {path}, shape = {embs.shape}")
    return embs

# ----------------------------
# 2) Assume your datasets are already loaded as `ds1` and `ds2`
#     with columns:  "image"  (PIL.Image or file path) and "question" (str)
# ----------------------------
# Example placeholder (remove these and pass your real datasets in):
# from datasets import Dataset
# ds1 = Dataset.from_dict({
#     "image": ["cat.jpg", "dog.jpg"],
#     "question": ["Describe the animal", "What is the dog doing?"]
# })
# ds2 = Dataset.from_dict({
#     "image": ["car.jpg", "plane.jpg"],
#     "question": ["What vehicle is this?", "Where is the plane?"]
# })

# >>> Replace ds1, ds2 with your actual datasets before running:
# ds1 = ...
# ds2 = ...
unsafe = MMSafetyBenchDataset(num_select=100)
ds1 = unsafe.get_dataset()

### load mini vqa as safe dataset
import pandas as pd
from datasets import Dataset
# file_path = "/root/data/train-00000-of-00001.parquet"
file_path = "/root/data/data-00000-of-00136.arrow"

df = pd.read_parquet(file_path)
df = df.rename(columns={"image": "image_path"})
print(f"Loaded dataframe with {len(df)} rows")
print(df.columns)  # 检查列名
columns = ["image_path", "question"]  # 根据你的数据修改
df = df[columns]
ds2 = Dataset.from_pandas(df)

os.makedirs("/root/data/tmp_images", exist_ok=True)

def save_bytes_to_file(example, idx):
    path = f"/root/data/tmp_images/{idx}.png"
    with open(path, "wb") as f:
        f.write(example["image_path"]["bytes"])
    example["image_path"] = path
    return example
ds2 = ds2.map(save_bytes_to_file, with_indices=True)

# safe = ShareGPT4vDataset(num_samples=500)
# ds2 = safe.get_dataset()
# breakpoint()

# ----------------------------
# 3) Compute embeddings
# ----------------------------
# embs1: (N1, D), embs2: (N2, D)
# If you use fp16 on CPU, cast to fp32 already handled by .float() before .cpu().numpy() above

from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# breakpoint()

# save_embeddings(embs1, save_dir="/root/mllm-safety-cot/logs", filename="mm_embs_100.npy")
embs1 = load_embeddings("/root/mllm-safety-cot/logs", "mm_embs_100.npy")
embs2 = embed_dataset(ds2, text_key="question", image_key="image_path")
save_embeddings(embs2, save_dir="/root/mllm-safety-cot/logs", filename="vqa_embs_100.npy")
embs2 = load_embeddings("/root/mllm-safety-cot/logs", "vqa_embs_100.npy")

print(embs1.shape)
print(embs2.shape)

# ----------------------------
# 4) t-SNE
# ----------------------------
X = np.vstack([embs1, embs2])  # (N1+N2, D)
labels = np.array([0] * len(embs1) + [1] * len(embs2))  # 0=DS1, 1=DS2

tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, min(30, (len(X) - 1) // 3)))
X2 = tsne.fit_transform(X)

# ----------------------------
# 5) Visualization (DS1=red, DS2=green)
# ----------------------------
colors = np.where(labels == 0, "red", "green")
plt.figure(figsize=(7, 7))
plt.scatter(X2[:, 0], X2[:, 1], c=colors, alpha=0.75, s=40, edgecolors="none")
plt.title("t-SNE of Multimodal (Image+Question) Embeddings\nUNSAFE=Red, SAFE=Green")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.tight_layout()
plt.show()

save_dir = "/root/mllm-safety-cot/logs"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "tsne_embeddings_qwen.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")