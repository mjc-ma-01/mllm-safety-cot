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
from dataset.base import MMSafetyBenchDataset,ShareGPT4v
import omegaconf
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
 ################
    # Create a data collator to encode text and image pairs
    ################
import wandb
wandb.init(project="sft_cot_")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
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
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    
    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    prompt_config_path = 'config/cot_prompt_llm.yaml'
    task_configs = omegaconf.OmegaConf.load(prompt_config_path)
    # dataset_mm = MMSafetyBenchDataset(task_configs=task_configs.mm_safetybench)
    # reject_cot_dataset = dataset_mm.get_cot_reject_dataset()
    
    dataset_mm = Dataset.load_from_disk("/fs-computility/ai-shen/majiachen/project/MLLM_CoT/dataset/cot_data_v1")
    dataset_mm = dataset_mm.map(lambda x: {"image_path": x["image"]})
    print(dataset_mm[0])
    reject_cot_dataset = dataset_mm.train_test_split(test_size=0.1)
    sharegpt4v = ShareGPT4v(num_samples=5000,cot=True)
    general_dataset = sharegpt4v.get_reject_dataset()
    
    print(len(reject_cot_dataset["train"]))
    print(len(reject_cot_dataset["test"]))
    print(len(general_dataset["train"]))
    print(len(general_dataset["test"]))
    
    # breakpoint()
    combined_train = concatenate_datasets([reject_cot_dataset["train"],general_dataset["train"]]).shuffle(seed=42)
    combined_test = concatenate_datasets([reject_cot_dataset["test"],general_dataset["test"]]).shuffle(seed=42)
    train_cot_dataset = DatasetDict({
        'train': combined_train,
        'test': combined_test
    }) 
    # train_cot_dataset = concatenate_datasets([reject_cot_dataset,general_dataset]).train_test_split(test_size=0.1)
    # breakpoint()
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
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = label_pad_token_id
        # if not dataset_args.train_on_prompt:
        prompt_len = prompt_tokens.shape[1]
        labels[0][:prompt_len] = label_pad_token_id
        prompt_response_batch["labels"] = labels
        return prompt_response_batch


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


