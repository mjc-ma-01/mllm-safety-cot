from dataset.base import *
from transformers import TextStreamer, MllamaForConditionalGeneration, AutoProcessor, HfArgumentParser,AutoModelForVision2Seq,AutoTokenizer, LlavaForConditionalGeneration,AutoModelForCausalLM
import omegaconf,torch,json
from PIL import Image
from dataclasses import dataclass
from datasets import Dataset
import tqdm,random
torch.manual_seed(1234)
rejection_templates = [
    "So, Sorry I cannot fulfill that request.",
    "Therefore, I am not able to provide an answer to this.",
    "So unfortunately, I cannot comply with that.",
    "So, I'm afraid I cannot answer that question.",
    "So, I cannot reveal that information.",
    "Therefore, I'm not able to disclose those details.",
    "Therefore, I can't share that with you.",
    "So, It's not safe to provide that information.",
    "In the end, I'm afraid I don't have access to that data.",
    "Therefore, this question is not appropriate for me to answer."
]
system_prompt: str = "You are a helpful and harmless assistant."
user_cot_prompt_template: str = (
    "You are given an image and a question, You should think step-by-step and answer the question.\n"
    "Question: {A}\n\n"
)
user_prompt_template: str = (
    "You are given an image and a question, please answer the question.\n"
    "Question: {A}\n\n"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen-VL-Chat
@dataclass
class ScriptArgument:
    model_path: str = "/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct"
    prompt_config_path: str  = './config/cot_prompt_llm.yaml'
    save_name: str = "cot_data_v2_risk"
    batch_size: int = 4

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArgument)
    args = parser.parse_args_into_dataclasses()[0]
    model = AutoModelForVision2Seq.from_pretrained(args.model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    # if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    model.to(device)
    task_configs = omegaconf.OmegaConf.load(args.prompt_config_path)
    
    dataset_ = NSFWDataset(task_configs=task_configs.mm_safetybench, think_mode=True)
    ds = dataset_.get_nsfw_cot_training_dataset()
    # ds = dataset_.get_nsfw_cot_training_dataset()
    
    print(ds[0])
    def multimodel_collator(examples): 
        messages = [[
                {"role":"user",
                 "content":[
                    {"type": "image", "text": None, "index": 0},
                    { "type":"text", "text": ex["prompt_question"], "index": None}
                    ] 
                 }] 
                    for ex in examples
            ]
        texts = [processor.apply_chat_template(message, tokenize = False, add_generation_prompt=True) for message in messages]
        images = [[Image.open(ex['image_path']).convert("RGB")] for ex in examples]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        return {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
     
    results = []
    from tqdm import tqdm
    bs = args.batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(ds), bs), desc="Generating answers in batch"):
            batch = ds.select(range(i, min(i + bs, len(ds))))
            inputs = multimodel_collator(batch)
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.8
                )
            result = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            for r in result:
                results.append(r)
                print(r)
                print("\n\n")

    
    save_log_path = f"./logs/cot_training_data/{args.save_name}.json"
    
    with open(save_log_path, "w") as file:
        final_results = [
            {
                "prompt_question": example["prompt_question"],
                "question": example["question"],
                "image_path": example["image_path"],
                "label": result + random.choice(rejection_templates),
                "category":example["category"]
            }
            for example, result in zip(ds, results)  
        ]
        json.dump(final_results, file, indent=4)
        
    dataset = Dataset.from_list(final_results)
    
    dataset.save_to_disk(f"./dataset/{args.save_name}")
    
    