# dev: WANDB_MODE=offline WANDB_PROJECT=system-prompt-steerability-dev PYTHONPATH=. srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=300 accelerate launch --config_file configs/accelerate_configs/single_gpu.yaml src/train.py
import os
import functools
from dataclasses import dataclass
import copy
import omegaconf
import torch
import transformers
from accelerate import PartialState
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, BitsAndBytesConfig, Trainer
from datasets import concatenate_datasets, Dataset
from data import *


@dataclass
class ModelArguments:
    model_path:       str = "/mnt/lustrenew/mllm_safety-shared/models/huggingface/HuggingFaceH4/zephyr-7b-beta"
    load_in_4bit:     bool = False
    use_flash_attention_2: bool = True
   
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
    output_dir: str = "/results/model:xxx/train:xxx"
    run_name: str = "/results/model:xxx/train:xxx"
    report_to: str = "wandb"
    overwrite_output_dir: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 4
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.05
    bf16: bool = True
    num_train_epochs: float = 2.0
    logging_steps: float = 10
    save_steps: float = 0.25
    eval_steps: float = 0.25
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, padding_side="right")
    # if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.unk_token
    
    # advbench = AdvBenchDataset(num_samples=1200,think_mode=True)
    # reject_cot_dataset = advbench.get_dataset()
    cot_data1 = DoNotAnswerDataset(num_samples=1200,think_mode=task_args.think_mode)
    ds1 = cot_data1.get_dataset()
    cot_data2 = HExDataset(num_samples=1200,think_mode=task_args.think_mode)
    ds2 = cot_data2.get_dataset()
    general_ds = GeneralDataset(num_samples=1200,think_mode=task_args.think_mode)
    general_dataset = general_ds.get_dataset()
    # breakpoint()
    combined_train = concatenate_datasets([ds1["train"],ds2["train"],general_dataset["train"]]).shuffle(seed=42)
    combined_test = concatenate_datasets([ds1["test"],ds2["test"],general_dataset["test"]]).shuffle(seed=42)
    train_dataset = DatasetDict({'train': combined_train,'test': combined_test})

    
    def sft_map(row, system_prompt: str = "You are a helpful and harmless assistant.",label_pad_token_id: int = -100) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["question"]},]
        
        prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        prompt_len = len(prompt_tokens)
        messages.append({"role": "assistant", "content": row["label"]})
        prompt_response_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
        labels = prompt_response_tokens.copy()
        labels[:prompt_len] = [label_pad_token_id] * prompt_len
        return {
            "input_ids": prompt_response_tokens,
            "attention_mask": [1]*len(prompt_response_tokens),
            "labels": labels,
        }
        
    train_dataset = train_dataset.map(sft_map)

    train_dataset = train_dataset.remove_columns(["label","question","harm_type","specific_harm_type"])
    print(f"-----------------------------{task_args.think_mode}--------------------------------")
    # initiate trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["test"],
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer,pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )
    # train and save
    trainer.train()
    save_name = "checkpoint-best" if training_args.load_best_model_at_end else "checkpoint-final"
    trainer.save_model(os.path.join(training_args.output_dir, save_name))


if __name__ == "__main__":
    train()
