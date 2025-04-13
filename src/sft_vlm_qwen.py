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
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration ,EarlyStoppingCallback
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from dataset.base import *
import omegaconf
from dataclasses import dataclass
from transformers import Qwen2VLProcessor
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
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, min_pixels=400*28*28, max_pixels=600*28*28)

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
    
    
    '''
    [ { "content":
        [ { "index": 0, "text": null, "type": "images" },
        { "index": null, "text": "\nWhat may be the purpose of this gathering in the field?", "type": "text" } ],
    "role": "user" }, 
    
    { "content": [
        { "index": null, "text": "The purpose of this gathering in the field is for a group of people to enjoy flying a giant lizard-shaped kite together. In the image, there are several individuals, along with the large kite dominating the scene. The lush green field provides an ample space for kite flying activities, allowing the participants to run and maneuver the kite in the open area. This event brings people together for recreational purposes, where they can bond and have fun while engaging in an outdoor activity.", "type": "text" } ], 
    "role": "assistant" } ]
    '''
    
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # breakpoint()
    combined_train = concatenate_datasets([ds["train"],general_ds["train"]]).shuffle(seed=42)
    combined_test = concatenate_datasets([ds["test"],general_ds["test"]]).shuffle(seed=42)
    train_cot_dataset = DatasetDict({
        'train': combined_train,
        'test': combined_test
    }) 

   
    def multimodal_collator(examples, label_pad_token_id=-100): 
        messages = []
        for example in examples:
            messages.append([
                {"content":[
                    {"text": None, "type": "image", "index": 0},
                    {"text": example["question"], "type":"text", "index": None}
                    ], 
                "role":"user"},
            ])
        prompt_texts = [processor.apply_chat_template(message, tokenize = False, add_generation_prompt=True) for message in messages]
        images = [Image.open(example['image_path']).convert("RGB") for example in examples]
        
        if isinstance(model, LlavaForConditionalGeneration):
            # LLava1.5 does not support multiple images
            images = [image[0] for image in images]
            
        prompt_batch = processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)
        prompt_tokens = prompt_batch['input_ids']
        for example, message in zip(examples, messages):
            message.append(
                {"content":[
                        {"text":example["label"], "type":"text", "index":None},
                    ], 
                 "role":"assistant"
                }
            )
           
        prompt_response_texts = [processor.apply_chat_template(message, tokenize = False, add_generation_prompt=False) for message in messages]
        
        prompt_response_batch = processor(text=prompt_response_texts, images=images, return_tensors="pt", padding=True)
        prompt_response_tokens = prompt_response_batch['input_ids']
        # breakpoint()
        
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = prompt_response_tokens.clone()
        labels[labels == processor.tokenizer.pad_token_id] = label_pad_token_id 
        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652,151653,151655]
        else: 
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
            
        # # if not dataset_args.train_on_prompt:
        # prompt_len = prompt_tokens.shape[1]
        # # labels[0][:prompt_len] = label_pad_token_id
        # labels[:,:prompt_len] = label_pad_token_id
        prompt_response_batch["labels"] = labels
        return prompt_response_batch
    
    
    from peft import LoraConfig
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM", 
    )
    args = SFTConfig(
    output_dir=training_args.output_dir, # directory to save and repository id
    num_train_epochs=3, 
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=5,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                       # push model to hub
    report_to="wandb",                # report metrics to tensorboard
    gradient_checkpointing_kwargs = {"use_reentrant": False}, # use reentrant checkpointing
    dataset_text_field="", # need a dummy field for collator
    dataset_kwargs = {"skip_prepare_dataset": True} # important for collator
)
    args.remove_unused_columns=False
    
    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=args,
        data_collator=multimodal_collator,
        train_dataset=train_cot_dataset[script_args.dataset_train_split],
        eval_dataset=train_cot_dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        peft_config=peft_config,
        # peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)


