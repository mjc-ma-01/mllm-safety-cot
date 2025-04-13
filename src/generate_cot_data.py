from dataset.base import MMSafetyBenchDataset,ShareGPT4v
from transformers import TextStreamer, MllamaForConditionalGeneration, AutoProcessor, HfArgumentParser,AutoModelForVision2Seq
import omegaconf,torch,json
from PIL import Image
from dataclasses import dataclass
from datasets import Dataset
import tqdm,random
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


@dataclass
class ScriptArgument:
    model_path: str = "/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct"
    prompt_config_path: str  = './config/cot_prompt.yaml'
    save_log_path: str = "./logs/cot_training_data/cot_data_v1.json"

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArgument)
    inference_args = parser.parse_args_into_dataclasses()[0]
    model = AutoModelForVision2Seq.from_pretrained(inference_args.model_path).to(device)
    processor = AutoProcessor.from_pretrained("/fs-computility/ai-shen/mllm_safety-shared/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct")
    task_configs = omegaconf.OmegaConf.load(inference_args.prompt_config_path)
   
    ## load dataset
    dataset_mm = MMSafetyBenchDataset(task_configs=task_configs.mm_safetybench)
    ds = dataset_mm.get_cot_training_dataset()
    print(ds[0])
    def multimodel_collator(example):
        messages = [
            [
                {"role": "user", 
                "content": [
                    {"type": "image", "text": None, "index": 0},
                    {"type": "text", "text": example["prompt_question"], "index": None}
                ]}
            ]
        ]
        input_text = processor.apply_chat_template(messages[0], add_generation_prompt=True, tokenize=False)
        image = Image.open(example["image_path"]).convert("RGB")
        inputs = processor(text=input_text, images=image, return_tensors="pt", padding=True).to(model.device)
        return inputs

    encoded_dataset = [multimodel_collator(example) for example in ds]
    results = []
    
    from tqdm import tqdm

    for i in tqdm(range(len(ds)), desc = "generating answers"):
        input_id = encoded_dataset[i]
        outputs = model.generate(
            **input_id,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.3
            )
        result =processor.decode(outputs[:, input_id["input_ids"].shape[-1]:][0],skip_special_tokens=True)
        results.append(result)
        # breakpoint()


    with open(inference_args.save_log_path, "w") as file:
        final_results = [
            {
                "prompt_question": example["prompt_question"],
                "question": example["question"],
                "image_path": example["image_path"],
                "label": result+" "+random.choice(rejection_templates),
                "catagory":example["catagory"]
            }
            for example, result in zip(ds, results)  
        ]
        json.dump(final_results, file, indent=4)
    # breakpoint()
    dataset = Dataset.from_list(final_results)
    dataset.save_to_disk("/fs-computility/ai-shen/majiachen/project/MLLM_CoT/dataset/cot_data_v1")
    # breakpoint()
    
    
    
    # with open("./logs/cot_training_data/results.json", "w") as file:
    #     final_results = [
    #         {
    #             "results" : results
    #         }
    #     ]  
    #     json.dump(final_results, file, indent=4)
        