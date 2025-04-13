# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    sft_vlm_smol_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path HuggingFaceTB/SmolVLM-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir sft-smol-vlm-hf \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_target_modules down_proj, o_proj, k_proj, q_proj, gate_proj, up_proj, v_proj

For LLaVA-NeXT, use: (requires transformers>=4.45)
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct
"""
from PIL import Image
from transformers import Qwen2VLProcessor, LlamaTokenizer,AutoTokenizer

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Idefics3ForConditionalGeneration,
    LlavaForConditionalGeneration,
)
from dataset.base import *

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
    ## important to model collator
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    think_mode = True
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
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    
    
    
    prompt_config_path = 'config/cot_prompt_llm.yaml'
    task_configs = omegaconf.OmegaConf.load(prompt_config_path)
    dataset_mm = MMSafetyBenchDataset(task_configs=task_configs.mm_safetybench,think_mode=think_mode)
    ds = dataset_mm.get_dataset()
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
            
        elif isinstance(model, Idefics3ForConditionalGeneration):
            image_tokens = [processor.tokenizer.additional_special_tokens_ids[
                processor.tokenizer.additional_special_tokens.index("<image>")
            ]]
        else: 
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    ## mask image tokens            
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch
    
    
    
    
    
    
    

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
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)