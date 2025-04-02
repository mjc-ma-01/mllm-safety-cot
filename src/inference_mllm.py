
import torch
import omegaconf
from datasets import Dataset
from torch.utils.data import DataLoader
from dataset.base import MMSafetyBenchDataset,MSSBenchDataset,ShareGPT4vDataset
from transformers import TextStreamer, MllamaForConditionalGeneration, AutoProcessor, HfArgumentParser,AutoModelForVision2Seq,PreTrainedTokenizer
from dataclasses import dataclass,field
from PIL import Image
from typing import Optional,Union,Literal
import os, json
from src.utils import *

@dataclass
class ScriptArgument:
    model_path: str = "/fs-computility/ai-shen/majiachen/results/sft_cot_llama+sharegpt4v02/checkpoint-2200"
    num_examples: str = 100
    prompt_config_path: str  = 'config/cot_prompt.yaml'
    save_log_path: str = "./logs/sft_answer/mix_cot_answer_v3_2200_sample_all.json"
    cot: bool = True
    test_dataset: str = "ood"
    ood_test_scene: str = "chat"
    rank: int = 0 ## 当前任务块
    world_size: int = 8 ## 总任务数量
    batch_size: int = 4 ## 推理时一个批次的数量
    # sample_ratio: float = 0.1 ## 测试集采样比例
    
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArgument)
    inference_args = parser.parse_args_into_dataclasses()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForVision2Seq.from_pretrained(inference_args.model_path)
    processor = AutoProcessor.from_pretrained("/fs-computility/ai-shen/mllm_safety-shared/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct")
    processor.tokenizer.padding_side = "left"
    
    # Wrap the model with DataParallel for multi-GPU usage
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    
    task_configs = omegaconf.OmegaConf.load(inference_args.prompt_config_path)
    
    # overrefusal test dataset
    mss_dataset = MSSBenchDataset(task_configs=task_configs.mm_safetybench,think_mode=inference_args.cot,scene=inference_args.ood_test_scene)
    # general test dataset
    sharegpt4v = ShareGPT4vDataset(num_samples=5000,think_mode=inference_args.cot)
    # IID safety test dataset
    iid_test_dataset = MMSafetyBenchDataset(task_configs=task_configs.mm_safetybench,think_mode=inference_args.cot)
    inference_args.num_examples = int(inference_args.num_examples) if inference_args.num_examples != "all" else "all"
    
    print(inference_args.num_examples)
    if inference_args.num_examples == "all":
        if inference_args.test_dataset == "ood":
            test_dataset = mss_dataset.get_reject_dataset()
            print(f"Dataset length: {len(test_dataset)}" if hasattr(test_dataset, "__len__") else "No length attribute")
            test_dataset = split_datasets(test_dataset,inference_args.rank, inference_args.world_size)
            print(len(test_dataset))
            print(test_dataset[0])
        elif inference_args.test_dataset == "general":
            test_dataset = sharegpt4v.get_reject_dataset()["test"]
            test_dataset = split_datasets(test_dataset,inference_args.rank, inference_args.world_size)
            print(len(test_dataset))
            print(test_dataset[0])
        elif inference_args.test_dataset == "iid":
            test_dataset = iid_test_dataset.get_reject_dataset()["test"]
            test_dataset = split_datasets(test_dataset,inference_args.rank, inference_args.world_size)
            print(len(test_dataset))
            print(test_dataset[0])
    else:
        if inference_args.test_dataset == "ood":
            test_dataset = mss_dataset.get_reject_dataset().select(range(inference_args.num_examples))
        elif inference_args.test_dataset == "general":
            test_dataset = sharegpt4v.get_reject_dataset()["test"].select(range(inference_args.num_examples))
            test_dataset = split_datasets(test_dataset,inference_args.rank, inference_args.world_size)
            print(len(test_dataset))
            print(test_dataset[0])
        elif inference_args.test_dataset == "iid":
            test_dataset = iid_test_dataset.get_reject_dataset()["test"].select(range(inference_args.num_examples))
  
    
    def multimodel_collator(examples): 
        messages = [[
                {"content":[
                    {"text": None, "type": "image", "index": 0},
                    {"text": ex["question"], "type":"text", "index": None}
                    ], "role":"user"}] for ex in examples
            ]
        prompt_texts = [processor.apply_chat_template(msg, tokenize = False, add_generation_prompt=True) for msg in messages]
        images = [[Image.open(ex['image_path']).convert("RGB")] for ex in examples]
        
        inputs = processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)
        return {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    
    ## 生成答案    
    results = []
    bs = inference_args.batch_size
    with torch.no_grad():
        for i in range(0, len(test_dataset), bs):
            batch = test_dataset.select(range(i, min(i + bs, len(test_dataset))))
            inputs = multimodel_collator(batch)
            # 生成文本答案
            outputs = model.module.generate(  # 注意这里要用 model.module
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.3)
            
            generated_texts = processor.batch_decode(outputs[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
            for ex, answer in zip(batch, generated_texts):
                results.append({
                    "question": ex["question"],
                    "image": ex["image_path"],
                    "answer": answer,
                    "label": ex.get("label", None)})
            print(f"num_{i}_examples")    
            print(results[i])

    # 保存结果
    save_path = inference_args.save_log_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"Results saved to {save_path}")
