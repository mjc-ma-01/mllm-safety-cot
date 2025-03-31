# dev: WANDB_MODE=offline WANDB_PROJECT=system-prompt-steerability-dev PYTHONPATH=. srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=300 accelerate launch --config_file configs/accelerate_configs/single_gpu.yaml src/train.py
import os
import functools
from dataclasses import dataclass

import omegaconf
import torch
import transformers
from accelerate import PartialState
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, BitsAndBytesConfig, Trainer
from datasets import concatenate_datasets, Dataset
from dataset.base import *
from dataset.base_llm import *


from utils import *
import src.utils as utils


@dataclass
class ModelArguments:
    model_path:       str = "/mnt/hwfile/llm-safety/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct"
    load_in_4bit:     bool = False
    use_flash_attention_2: bool = True

@dataclass
class TaskArguments:
    task_config_path: str  = "configs/tasks/imdb_preference.yaml"
    train_task_names: str  = "AdvBench+general_v0"
    augment:          bool = True
    num_proc:         int  = 8

@dataclass
class PeftArguments:
    use_peft:       bool  = True
    target_modules: str   = "all-linear"
    r:              int   = 64
    lora_alpha:     int   = 64
    lora_dropout:   float = 0.05
    bias:           str   = "none"
    task_type:      str   = "CAUSAL_LM"

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = "/mnt/lustrenew/mllm_safety-shared/majaichen/results/model:xxx/train:xxx"
    run_name: str = "mnt/lustrenew/mllm_safety-shared/majaichen/results/model:xxx/train:xxx"
    report_to: str = "wandb"
    overwrite_output_dir: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.0
    weight_decay: float = 0.05
    bf16: bool = False
    num_train_epochs: float = 3.0
    logging_steps: float = 10
    save_steps: float = 0.1
    eval_steps: float = 0.1
    eval_strategy: str = "steps"
    save_only_model: bool = False
    load_best_model_at_end: bool = True


def train():
    parser = transformers.HfArgumentParser((ModelArguments, PeftArguments, TaskArguments, TrainingArguments))
    model_args, peft_args, task_args, training_args = parser.parse_args_into_dataclasses()

    # loading model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,
        **(
            {"device_map": {"": PartialState().local_process_index}}
            if not transformers.modeling_utils.is_deepspeed_zero3_enabled()
            else {}
        ),
        quantization_config=(
            BitsAndBytesConfig(load_in_4bit=True)
            if model_args.load_in_4bit and transformers.utils.is_bitsandbytes_available()
            else None
        ),
        attn_implementation=(
            "flash_attention_2"
            if model_args.use_flash_attention_2 and transformers.utils.is_flash_attn_2_available()
            else None
        ),
    )    
        
    ################################################################
    ####################### # Dataset################################
    ################################################################
    prompt_config_path = 'config/cot_prompt_llm.yaml'
    task_configs = omegaconf.OmegaConf.load(prompt_config_path)
    # dataset_mm = Dataset.load_from_disk("/fs-computility/ai-shen/majiachen/project/MLLM_CoT/dataset/cot_data_v1")
    
    dataset_adv = AdvBenchDataset(task_configs=task_configs.advbench,num_samples=2000,cot=True)
    reject_cot_dataset = dataset_adv.get_reject_dataset()
    dataset_gen = GeneralDataset(num_samples=5000,cot=True)
    general_dataset = dataset_gen.get_reject_dataset()
    
    print(len(reject_cot_dataset["train"]))
    print(len(reject_cot_dataset["test"]))
    print(len(general_dataset["train"]))
    print(len(general_dataset["test"]))
    
    # breakpoint()
    combined_train = concatenate_datasets([reject_cot_dataset["train"],general_dataset["train"]]).shuffle(seed=42)
    combined_test = concatenate_datasets([reject_cot_dataset["test"],general_dataset["test"]]).shuffle(seed=42)
    train_dataset = DatasetDict({
        'train': combined_train,
        'test': combined_test
    })
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, padding_side="right")
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    breakpoint()
    
    def sft_map(
        row, 
        tokenizer: PreTrainedTokenizer, 
        train_on_prompt: bool = False, 
        label_pad_token_id: int = -100
    ) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": row["question"]
            },
        ]
        prompts = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True
        )
        prompt_tokens = tokenizer.encode(prompts, add_special_tokens=False)
        messages.append({"role": "assistant", "content": row["label"]})
        prompt_response = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=False, 
            tokenize=True
        )
        prompt_response_tokens = tokenizer.encode(prompt_response, add_special_tokens=False)
        assert prompt_response_tokens[-1] != tokenizer.eos_token_id
        prompt_response_tokens = prompt_response_tokens + [tokenizer.eos_token_id]
        labels = prompt_response_tokens.copy()
        
        if not train_on_prompt:
            prompt_len = len(prompt_tokens)
            labels[:prompt_len] = [label_pad_token_id] * prompt_len
        return {
            "input_ids": prompt_response_tokens,
            "attention_mask": [1]*len(prompt_response_tokens),
            "labels": labels,
        }


    train_dataset = train_dataset.map(
        functools.partial(sft_map, tokenizer=tokenizer), 
        num_proc=task_args.num_proc)

    # initiate trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["test"],
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt", 
            padding=True
        )
    )
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-0"))
    # train and save
    trainer.train()
    save_name = "checkpoint-best" if training_args.load_best_model_at_end else "checkpoint-final"
    trainer.save_model(os.path.join(training_args.output_dir, save_name))


if __name__ == "__main__":
    train()
