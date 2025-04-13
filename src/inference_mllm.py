
import torch
import omegaconf
from datasets import Dataset
from torch.utils.data import DataLoader
from dataset.base import MMSafetyBenchDataset,MSSBenchDataset,ShareGPT4vDataset
from transformers import TextStreamer, MllamaForConditionalGeneration, AutoProcessor, HfArgumentParser,AutoModelForVision2Seq,PreTrainedTokenizer,AutoModelForImageTextToText
from dataclasses import dataclass,field
from PIL import Image
from typing import Optional,Union,Literal
import os, json
from src.utils import *
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_map={
    "pixtral_12b": "/mnt/lustrenew/mllm_safety-shared/models/huggingface/mistralai/Pixtral-12B-2409",
    "gamma3_4b": "//mnt/lustrenew/mllm_safety-shared/models/huggingface/google/gemma-3-4b-it",
    "SmolVLM": "/mnt/lustrenew/mllm_safety-shared/models/huggingface/HuggingFaceTB/SmolVLM-Instruct",
    "llava1.5_7b": "/mnt/lustrenew/mllm_safety-shared/models/huggingface/llava-hf/llava-v1.6-mistral-7b-hf",
    "qwen_7b_vl": "/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2-VL-7B-Instruct",
    "llama_11b": "/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct"
}


@dataclass
class ScriptArgument:
    use_peft: bool = True
    model_identifier: str = "qwenvl_7b"
    model_path: str = "/mnt/lustrenew/mllm_safety-shared/tmp/majiachen/results/model:sft_mllm_qwenvl_7b/train:mmsafetybench+sharedgpt4v_v1"
    prompt_config_path: str  = 'config/cot_prompt_llm.yaml'
    save_log_path: str = "./logs/sft_answer/test.json"
    cot: bool = True
    test_dataset: str = "ood"
    rank: int = 0 ## 当前任务块
    world_size: int = 8 ## 总任务数量
    batch_size: int = 1 ## 推理时一个批次的数量
    
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArgument)
    src_args = parser.parse_args_into_dataclasses()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = model_map[src_args.model_identifier]
    
    if src_args.use_peft and src_args.model_identifier != "gamma3_4b":
        adapter_model = src_args.model_path
        base_model = AutoModelForVision2Seq.from_pretrained(base_model_path)
        model = PeftModel.from_pretrained(base_model, adapter_model)
        
    elif src_args.model_identifier == "gamma3_4b":
        adapter_model = src_args.model_path
        base_model = AutoModelForImageTextToText.from_pretrained(src_args.model_path)
        model = PeftModel.from_pretrained(base_model, adapter_model)
        
    else:
        model = AutoModelForVision2Seq.from_pretrained(src_args.model_path)
        
    processor = AutoProcessor.from_pretrained(base_model_path,trust_remote_code=True)
    if src_args.model_identifier == "qwenvl_7b":
        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True, min_pixels=400*28*28, max_pixels=600*28*28)
        processor.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    processor.tokenizer.padding_side = "left"
    
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    
    task_configs = omegaconf.OmegaConf.load(src_args.prompt_config_path)
    ds_mss = MSSBenchDataset(task_configs=task_configs.mm_safetybench,think_mode=src_args.cot)
    ds_share = ShareGPT4vDataset(num_samples=5000,think_mode=src_args.cot)
    ds_mm = MMSafetyBenchDataset(task_configs=task_configs.mm_safetybench,think_mode=src_args.cot)
    
    if src_args.test_dataset == "ood":
        test_dataset = ds_mss.get_dataset()
    elif src_args.test_dataset == "general":
        test_dataset = ds_share.get_dataset()["test"]
    elif src_args.test_dataset == "iid":
        test_dataset = ds_mm.get_dataset()["test"]
            
            
    print(f"Dataset length: {len(test_dataset)}" if hasattr(test_dataset, "__len__") else "No length attribute")
    test_dataset = split_datasets(test_dataset,src_args.rank, src_args.world_size)
    print(len(test_dataset))
    print(test_dataset[0])
    
    
    def multimodel_collator(examples): 
        messages = [[
                {"role":"user",
                 "content":[
                    {"type": "image", "text": None, "index": 0},
                    { "type":"text", "text": ex["question"], "index": None}
                    ] 
                 }] 
                    for ex in examples
            ]
        texts = [processor.apply_chat_template(message, tokenize = False, add_generation_prompt=True) for message in messages]
        images = [[Image.open(ex['image_path']).convert("RGB")] for ex in examples]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        return {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    
    ## 生成答案    
    results = []
    bs = src_args.batch_size
    with torch.no_grad():
        for i in range(0, len(test_dataset), bs):
            batch = test_dataset.select(range(i, min(i + bs, len(test_dataset))))
            inputs = multimodel_collator(batch)
            outputs = model.module.generate(  
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7)
            
            generated_texts = processor.batch_decode(outputs[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
            for ex, answer in zip(batch, generated_texts):
                results.append({
                    "question": ex["question"],
                    "image": ex["image_path"],
                    "answer": answer,
                    "label": ex.get("label", None)})
            print(f"num_{i}_examples")    
            print(results[i])

    save_path = src_args.save_log_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"Results saved to {save_path}")
