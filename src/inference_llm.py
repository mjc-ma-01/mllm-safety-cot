
import torch
import omegaconf
from datasets import Dataset
from torch.utils.data import DataLoader
from dataset.base import MMSafetyBenchDataset,MSSBenchDataset,ShareGPT4vDataset
from dataset.base_llm import *
from transformers import TextStreamer, MllamaForConditionalGeneration, AutoProcessor, HfArgumentParser,AutoModelForVision2Seq,PreTrainedTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,AutoTokenizer
from dataclasses import dataclass,field
from PIL import Image
from typing import Optional,Union,Literal
import os, json
from src.utils import *

model_name="sft_llm_zephyer"
train_task_names="HEX_270+DoNotAns_720+ultrachat_1200_v1"

@dataclass
class ScriptArgument:
    model_path: str = f"/mnt/lustrenew/mllm_safety-shared/tmp/majiachen/results/model:{model_name}/train:{train_task_names}/checkpoint-best"
    # model_path: str = "/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct"
    # model_path: str = "/mnt/lustrenew/mllm_safety-shared/models/huggingface/google/zephyr-7b-beta"
    num_examples: str = "all"
    save_log_path: str = f"./logs/sft_answer/model:{model_name}/train:{train_task_names}/test:ood/test.json"
    cot: bool = True
    test_dataset: str = "iid"
    rank: int = 0 ## 当前任务块
    world_size: int = 8 ## 总任务数量
    batch_size: int = 4 ## 推理时一个批次的数量
    
    load_in_4bit: bool = False
    use_flash_attention_2: bool = True
    batch_size: int = 4 
    seed: int = 42
    # sample_ratio: float = 0.1 ## 测试集采样比例
    
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArgument)
    inference_args = parser.parse_args_into_dataclasses()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformers.set_seed(inference_args.seed)
    # loading model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        inference_args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=(
            BitsAndBytesConfig(load_in_4bit=True)
            if inference_args.load_in_4bit and transformers.utils.is_bitsandbytes_available()
            else None
        ),
        attn_implementation=(
            "flash_attention_2"
            if inference_args.use_flash_attention_2 and transformers.utils.is_flash_attn_2_available()
            else None
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(inference_args.model_path, padding_side="left")
    # if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.unk_token
    
    if inference_args.num_examples == "all":
        # overrefusal test dataset
        if inference_args.test_dataset == "ood":
            # test_dataset = XSTest(think_mode=inference_args.cot)
            test_dataset = StrongREJECT(think_mode=inference_args.cot)
            
        # general test dataset
        elif inference_args.test_dataset == "general":
            test_dataset = GeneralDataset(num_samples=1200,think_mode=inference_args.cot)
        # IID safety test dataset
        elif inference_args.test_dataset == "iid":
            test_dataset = DoNotAnswerDataset(num_samples=1200,think_mode=inference_args.cot)

    test_dataset = test_dataset.get_dataset()["test"]
    print(f"{inference_args.test_dataset} Dataset length: {len(test_dataset)}" )
    test_dataset = split_datasets(test_dataset,inference_args.rank, inference_args.world_size)
    print(f"{inference_args.test_dataset} subset length:",len(test_dataset))
    print(f"{inference_args.test_dataset} subset demo:",test_dataset[0])
    # breakpoint()         
    def llm_collator(row, system_prompt: str="You are a helpful and harmless assistant.") -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            { "role": "user", "content": row["question"]},]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return {"prompt": prompt}

    test_dataset = test_dataset.map(llm_collator)
 
 ## 生成答案
    results = []
    bs = inference_args.batch_size
    for i in range(0, len(test_dataset), bs):
        batch = test_dataset.select(range(i, min(i + bs, len(test_dataset))))
        prompts = batch["prompt"]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        # 生成文本答案
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=500,  # 控制最大生成长度
                num_return_sequences=1,  # 生成一个答案
                do_sample=True,  # 采用贪心解码（可以改成True用于采样）
                temperature=0.3,  # 控制生成的多样性（仅在 do_sample=True 时有效）
            )
        generated_texts = tokenizer.batch_decode(generated_ids[:, input_ids.shape[-1]:], skip_special_tokens=True) 
        for ex, answer in zip(batch, generated_texts):
            results.append({
                "question": ex["question"],
                "answer": answer,
                "label": ex.get("label", None)})
        print(f"num_{i}_examples")
        # breakpoint()    
        print(results[i])

    # 保存结果
    save_path = inference_args.save_log_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"Results saved to {save_path}")
