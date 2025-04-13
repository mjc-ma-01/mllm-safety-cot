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
smolVLM\llama\gamma3 model refer to trl sft official repo 

Train smolVLM .
## reference to https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_smol_vlm.py

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

    
Train Llama .
## reference to https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py

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
    

Train Gemma-3 on the HuggingFaceH4/llava-instruct-mix-vsft dataset (single-image).
## reference to https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_gemma3.py

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3-4b-it-trl-sft-ChartQA \
    --bf16 \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager



qwen and llava model refered to bolg: https://www.philschmid.de/fine-tune-multimodal-llms-with-trl


"""
from dataclasses import dataclass
from PIL import Image
from transformers import Qwen2VLProcessor, LlamaTokenizer,AutoTokenizer,AutoModelForImageTextToText,Gemma3ForConditionalGeneration

import torch
from datasets import load_dataset,load_from_disk
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
model_map={
    "pixtral_12b": "/mnt/lustrenew/mllm_safety-shared/models/huggingface/mistralai/Pixtral-12B-2409",
    "gamma3_4b": "/mnt/lustrenew/mllm_safety-shared/models/huggingface/google/gemma-3-4b-it",
    "SmolVLM": "/mnt/lustrenew/mllm_safety-shared/models/huggingface/HuggingFaceTB/SmolVLM-Instruct",
    "llava1.5_7b": "/mnt/lustrenew/mllm_safety-shared/models/huggingface/llava-hf/llava-v1.6-mistral-7b-hf",
    "qwen_7b_vl": "/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2-VL-7B-Instruct",
    "llama_11b": "/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct"
}

@dataclass
class MyArguments:
    version: str = "v1"
    think_mode: bool = True
    model_identifier: str = "gamma3_4b"


if __name__ == "__main__":
    parser = TrlParser(dataclass_types=(ScriptArguments, SFTConfig, ModelConfig, MyArguments))
    script_args, training_args, model_args, args = parser.parse_args_and_config()
    # parser = TrlParser(dataclass_types=[MyArguments])
    # training_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}


    model_args.model_name_or_path = model_map[args.model_identifier]

    
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
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)
    
    if args.model_identifier == "qwenvl_7b":
        processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, min_pixels=400*28*28, max_pixels=600*28*28)
        
    if args.model_identifier == "llava1.5_7b":
        processor.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        
    # processor.tokenizer.padding_side= "right"
    
    if args.model_identifier == "gamma3_4b":
        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs)
    
    
    
    prompt_config_path = 'config/cot_prompt_llm.yaml'
    task_configs = omegaconf.OmegaConf.load(prompt_config_path)
    
    ## v1 data
    if args.version == "v1":
        dataset_mm = MMSafetyBenchDataset(task_configs=task_configs.mm_safetybench,think_mode=args.think_mode)
        ds = dataset_mm.get_dataset()
        
    ## v2 data
    elif args.version == "v2":
        # with open("./logs/cot_training_data/cot_data_v1.json", "r") as f:
        #     data = json.load(f)
        # dataset = Dataset.from_list(data).remove_columns(["prompt_question","catagory"])
        # dataset.save_to_disk("./dataset/cot_data_v1")
        ds = load_from_disk("./dataset/cot_data_v1")
        ds = ds.train_test_split(test_size=0.1)
        
    sharegpt4v = ShareGPT4vDataset(num_samples=5000,think_mode=args.think_mode)
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
    
    def multimodal_collator(examples, label_pad_token_id=-100): 
        messages = []
        for example in examples:
            messages.append([
                {"content":[
                    {"text": None, "type": "image", "index": 0},
                    {"text": example["question"], "type":"text", "index": None}
                    ], "role":"user"},
            ])
        prompt_texts = [processor.apply_chat_template(message, tokenize = False, add_generation_prompt=True) for message in messages]
        images = [Image.open(example['image_path']).convert("RGB") for example in examples]
        prompt_batch = processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)
        prompt_tokens = prompt_batch['input_ids']
        for example, message in zip(examples, messages):
            message.append(
                {"content":[
                        {"text":example["label"], "type":"text", "index":None},
                    ], "role":"assistant"
                    }
            )
        prompt_response_texts = [processor.apply_chat_template(message, tokenize = False, add_generation_prompt=False) for message in messages]
        prompt_response_batch = processor(text=prompt_response_texts, images=images, return_tensors="pt", padding=True)
        prompt_response_tokens = prompt_response_batch['input_ids']
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = prompt_response_tokens.clone()
        labels[labels == processor.tokenizer.pad_token_id] = label_pad_token_id 
        # Ignore the image token index in the loss computation (model specific)
                
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652,151653,151655]
            
        elif isinstance(model, Idefics3ForConditionalGeneration):
    ## for smol model
            image_tokens = [processor.tokenizer.additional_special_tokens_ids[
                processor.tokenizer.additional_special_tokens.index("<image>")
            ]]
        elif isinstance(model, Gemma3ForConditionalGeneration):
    ## set '<image_soft_token>' and <start_of_image>' to padding for gamma3 model
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["boi_token"])]
            labels[labels == 262144] = -100
        else: 
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
            
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
            
        # if not dataset_args.train_on_prompt:
        prompt_len = prompt_tokens.shape[1]
        labels[0][:prompt_len] = label_pad_token_id
        prompt_response_batch["labels"] = labels
        return prompt_response_batch

    # train_dataset = train_cot_dataset.map(format_data, batched=True, batch_size=4)


    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=multimodal_collator,
        train_dataset=train_cot_dataset[script_args.dataset_train_split],
        eval_dataset=train_cot_dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
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