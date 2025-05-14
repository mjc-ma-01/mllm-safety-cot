
import torch
import omegaconf
from datasets import Dataset
from torch.utils.data import DataLoader
from collections.abc import Sequence

# from flash_attn import FlashAttention
from dataset.base import MMSafetyBenchDataset,MSSBenchDataset,ShareGPT4vDataset,SIUO
from transformers import TextStreamer, MllamaForConditionalGeneration, AutoProcessor, HfArgumentParser,AutoModelForVision2Seq,PreTrainedTokenizer,AutoModelForImageTextToText
from dataclasses import dataclass,field
from typing import Optional,Union,Literal
import os, json
from src.utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer
# PYTHONPATH=. srun -p mllm_safety --quotatype=spot --gres=gpu:4 --cpus-per-task=16 --time=3000 python src/generate_template.py
model_map={
    "llama_11b": "/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct",
}
safety_categories: Sequence[str] = [
    'physical harm', 'economicharm', 'health consultation', 'legal opinion',
    'fraud', 'gov decision', 'privacy violence', 'malware generation',
    'political lobbying', 'financial advice', 'sex', 'hatespeech',
    'illegal activity', 'bloody', 'hentai', 'porn', 'protest', 'sexy',
    'alcohol', 'cigarette', 'gun', 'insulting gesture', 'knife'
]

# generate_prompt: str = """"
#     You are an AI assistant trained to generate refusal statements that explain clearly, sincerely, and concretely why a question or image related to a specific safety category cannot be answered. Your goal is to help users understand the risks involved and why such content should be avoided.

#     Please generate 15 distinct refusal statements for the specific safety category: {A}. Each statement should:
#     - Describe why this image related to this category may be harmful or dangerous.
#     - Mention potential consequences (e.g., physical, psychological, legal, financial).
#     - Be professional, empathetic, and concise.
#     - Avoid generic phrasing like "against the rules" and instead give grounded, realistic reasoning.
#     - Vary in wording and sentence structure.
#     **Important**: Please directly output the 15 refusal statements in plain text format, one per line. **Do not** include any headings, titles, explanations, or extra content.
# #     Now, generate 15 refusal statements for the category: {A}."""
# generate_prompt = """You are an AI assistant trained to generate refusal statements that clearly, sincerely, and concretely explain why a question or image related to a specific safety category cannot be answered. Your goal is to help users understand the potential risks and harmful consequences of such content.

# Please generate 15 distinct refusal statements for the safety category: {A}. Each statement should begin with phrase like "This image may inolves {A}". Each statement should:
# - Mention potential consequences such as physical injury, psychological trauma, legal implications, or financial loss.
# - Explain why the content may be dangerous or inappropriate.
# - Avoid vague phrases like "against the rules"; instead, use realistic and grounded reasoning.
# - Vary sentence structure and vocabulary to ensure diversity.
# - Be professional, concise, and empathetic.

# **Important**: Please directly output the 15 refusal statements in plain text format, one per line. **Do not** include any headings, titles, explanations, or extra content.

# Now, generate 15 refusal statements for the category: {A}."""

generate_prompt = """You are an AI assistant that writes refusal statements to explain why certain images should not be described or discussed.

Please generate 15 detailed and diverse refusal statements for the safety category: {A}.  
Each statement should:

- Start with a phrase like: "This image may involve {A}" or "This image may depict..."  
- Clearly describe what the image shows that relates to this category.  
- Explain why this content is harmful, dangerous, or inappropriate, with realistic consequences such as physical harm, psychological impact, legal risks, or financial damage.  
- Avoid vague or generic phrases like “against the rules”; instead, provide concrete reasoning.  
- Be professional, empathetic, and concise, with varied sentence structure and vocabulary.

**Important**: Please directly output the 15 refusal statements in plain text format, one per line. **Do not** include any headings, titles, explanations, or extra content.

Now, generate 15 refusal statements for the category: {A}."""





    # For example, if the category is "Financial Harm," your responses might include statements about scams, unethical financial practices, or fraudulent schemes and why discussing them is risky."""
    
@dataclass
class ScriptArgument:
    use_peft: bool = True
    model_identifier: str = "llama_11b"
    prompt_config_path: str  = 'config/cot_prompt_llm.yaml'
    save_log_path: str = "./logs/cot_training_data/cot_template_1.yaml"
    batch_size: int = 4 ## 推理时一个批次的数量
    
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArgument)
    src_args = parser.parse_args_into_dataclasses()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_map[src_args.model_identifier]
    
     
    model = AutoModelForVision2Seq.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    
    annotations = []
    for i in safety_categories:
        annotations.append({
            "question": generate_prompt.format(A=i),
            "categories": i
        }
        )
    test_dataset = Dataset.from_list(annotations)
    print(f"Dataset length: {len(test_dataset)}" if hasattr(test_dataset, "__len__") else "No length attribute")
    print(len(test_dataset))
    print(test_dataset[0])
    
    
    def multimodel_collator(examples): 
        messages = [[
                {"role":"user",
                 "content":[
                    # {"type": "image", "text": None, "index": 0},
                    { "type":"text", "text": ex["question"], "index": None}
                    ] 
                 }] 
                    for ex in examples
            ]
        texts = [processor.apply_chat_template(message, tokenize = False, add_generation_prompt=True) for message in messages]
        # images = [[Image.open(ex['image_path']).convert("RGB")] for ex in examples]
        
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        return {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    
    ## 生成答案    
    # results = []
    from collections import defaultdict
    import yaml
    results = defaultdict(list)
    
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
                category = ex["categories"]
                # 如果每次生成的是一整段包含多条陈述，用换行分开成 list
                statements = [s.strip() for s in answer.strip().split("\n") if s.strip()]
                results[category].extend(statements)
            # breakpoint()    
            print(f"num_{i}_examples")    
            
    breakpoint()
    save_path = src_args.save_log_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(dict(results), f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # with open(save_path, "w", encoding="utf-8") as file:
    #     json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"Results saved to {save_path}")
