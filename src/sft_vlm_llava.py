# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
pip install pillow

# Tested on 8x H100 GPUs
srun -p mllm-align --quotatype=reserved --gres=gpu:4 --cpus-per-task=16 --time=3000 \
    accelerate launch --config_file=config/deepspeed_zero3.yaml \
    sft_vlm.py \
    --dataset_name aa \
    --model_name_or_path /mnt/hwfile/llm-safety/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir sft-Llama-3.2-11B-Vision-Instruct \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing

For LLaVA-NeXT, use: (requires transformers>=4.45)
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct
"""
from PIL import Image
import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration ,EarlyStoppingCallback,LlavaNextProcessor
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from dataset.base import *
import omegaconf
from dataclasses import dataclass
from transformers import Qwen2VLProcessor, LlamaTokenizer,AutoTokenizer
from qwen_vl_utils import process_vision_info
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    think_mode = True
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}


    # You MUST put the below items for vision finetuning:
    training_args.dataset_text_field = ""
    # training_args.dataset_num_proc = 4
    training_args.max_seq_length = 2048
    
    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    if model_args.model_name_or_path.split("/")[-1] == "Qwen2-VL-7B-Instruct":
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, min_pixels=400*28*28, max_pixels=600*28*28)
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)
        processor.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        processor.tokenizer.padding_side="right"
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    ################
    # Dataset
    ################
    prompt_config_path = 'config/cot_prompt_llm.yaml'
    task_configs = omegaconf.OmegaConf.load(prompt_config_path)
    dataset_mm = MMSafetyBenchDataset(task_configs=task_configs.mm_safetybench,think_mode=think_mode)
    ds = dataset_mm.get_dataset()
    
    # dataset_mm = Dataset.load_from_disk("/mnt/petrelfs/majiachen/project/mllm-safety-cot/dataset/cot_data_v1")
    # dataset_mm = dataset_mm.map(lambda x: {"image_path": x["image"]})
    # print(dataset_mm[0])
    # ds = dataset_mm.train_test_split(test_size=0.1)
    
    sharegpt4v = ShareGPT4vDataset(num_samples=5000,think_mode=think_mode)
    general_ds = sharegpt4v.get_dataset()
    print(len(ds["train"]))
    print(len(ds["test"]))
    print(len(general_ds["train"]))
    print(len(general_ds["test"]))
    combined_train = concatenate_datasets([ds["train"],general_ds["train"]]).shuffle(seed=42)
    combined_test = concatenate_datasets([ds["test"],general_ds["test"]]).shuffle(seed=42)
    train_cot_dataset = DatasetDict({
        'train': combined_train,
        'test': combined_test
    }) 
    
    def format_data(batch):
        messages = []
        for i in range(len(batch["question"])):
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "text": None, "index": 0},
                        {"type": "text", "text": batch["question"][i], "index": None}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": batch["label"][i], "index": None}
                    ]
                }
            ]
            messages.append(message)
        return {"messages": messages}
        
  
    def multimodal_collator(examples, label_pad_token_id=-100): 
        texts = [processor.apply_chat_template(example["messages"], tokenize = False) for example in examples]
        images = [Image.open(example['image_path']).convert("RGB") for example in examples]
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch['input_ids'].clone()
    ## mask pad tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652,151653,151655]
        else: 
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    ## mask image tokens            
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch
    
## reference to https://www.philschmid.de/fine-tune-multimodal-llms-with-trl 
    from peft import LoraConfig
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    if model_args.use_peft == True:
        peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.05,
                r=8,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM", 
        )
    else:
        peft_config = None
    train_dataset = train_cot_dataset.map(format_data,batched=True,batch_size=4)
    
    ################
    # Training
    ################


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=multimodal_collator,
        train_dataset=train_dataset[script_args.dataset_train_split],
        eval_dataset=train_dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)


