from datasets import Dataset, DatasetDict
from typing import Optional, Sequence, List, Dict
import os
from datasets import load_dataset, concatenate_datasets
import json, random
import transformers
from pathlib import Path
import omegaconf
from tqdm import tqdm
from datasets import Dataset, load_dataset
import pandas as pd
import os
from datasets import load_from_disk
# srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 python dataset/base.py
reject_templates = [
    "Sorry, I cannot fulfill that request.",
    "I am not able to provide an answer to this.",
    "Unfortunately, I cannot comply with that.",
    "I'm afraid I cannot answer that question.",
    "I cannot reveal that information.",
    "I'm not able to disclose those details.",
    "I cannot share that with you.",
    "It is not safe to provide that information.",
    "I'm afraid I don't have access to that data.",
    "This question is not appropriate for me to answer."
]
reject_cot_responses = [
    " Therefore, I'm sorry, but I am unable to assist with that request.",
    " Thus, my apologies, but I can't comply with this request.",
    " So, unfortunately, I can't fulfill your request at the moment.",
    " For this reason, I'm sorry, but I cannot help with this request.",
    " As a result, my apologies, but I am unable to process that request.",
    " Because of this, sorry, but I'm not able to comply with this request.",
    " Due to these concerns, I'm sorry, but I can't provide that information.",
    " Consequently, my apologies, but I cannot provide assistance with this.",
    " So, sorry, but I cannot fulfill this request.",
    " So, I'm sorry, but I'm unable to help with this request."
]

system_prompt: str = "You are a helpful and harmless assistant."
user_cot_prompt_template: str = (
    "You are given a question, you should think step-by-step and answer the question.\n"
    "Question: {A}\n\n"
)
user_prompt_template: str = (
    "You are given a question, you should answer the question.\n"
    "Question: {A}\n\n"
)

generate_cot_prompt_template: str = (
    "You are given a question. Please provide a very brief explanation why it may potentially cause {A}, considering the user intent.\n"
    "Question: {B}\n\n"
)
     

    
  
# class GeneralDataset(Dataset):
#     think_mode: bool  = True
#     def __init__(self,num_samples,think_mode,subset) -> None:
#         self.subset = subset
#         self.num_samples = num_samples
#         self.think_mode = think_mode
#         if self.subset == "full":
#             '''
#             helpful-base: train:43835,test:2354
#             harmless-base: train:42537,test:2312
#             '''
#             dataset_helpful  = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
#             dataset_harmless = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
#             combined_train = concatenate_datasets([dataset_helpful['train'], dataset_harmless['train']])
#             combined_test = concatenate_datasets([dataset_helpful['test'], dataset_harmless['test']])
#             self.dataset = DatasetDict({
#                 'train': combined_train,
#                 'test': combined_test
#             })
#     def get_reject_dataset(self):
#         def extract_single_turn_dialogue(input_text):
#             """
#             提取单轮对话，将 Human 映射为 prompt，Assistant 映射为 label，过滤掉多轮对话。
#             """
#             lines = input_text.strip().split('\n\n')
#             dialogue = {}

#             if len(lines) == 2:  # 只保留一轮对话
#                 for line in lines:
#                     if line.startswith("Human:"):
#                         dialogue["prompt"] = line.replace("Human: ", "").strip()
#                     elif line.startswith("Assistant:"):
#                         dialogue["label"] = line.replace("Assistant: ", "").strip()
#                 return dialogue
#             else: return None
        
#         train_dataset = self.dataset["train"].map(
#             lambda row :{
#                 "chosen": extract_single_turn_dialogue(row["chosen"])
#             }
#         ).filter(lambda row: row["chosen"]!=None).map(
#             lambda row:{
#             "question":user_cot_prompt_template.format(A=row["chosen"]["prompt"]) if self.think_mode else user_prompt_template.format(A=row["chosen"]["prompt"]),
#             "label":row["chosen"]["label"]}
#             ).remove_columns(["chosen","rejected"]).shuffle(seed=42).select(range(self.num_samples)).train_test_split(test_size=0.1)
        
        
#         test_dataset = self.dataset["test"].map(
#             lambda row :{
#                 "chosen": extract_single_turn_dialogue(row["chosen"])
#             }
#         ).filter(lambda row: row["chosen"]!=None).map(
#             lambda row:{
#             "question":user_cot_prompt_template.format(A=row["chosen"]["prompt"]) if self.think_mode else user_prompt_template.format(A=row["chosen"]["prompt"]),
#             "label":row["chosen"]["label"]}
#             ).remove_columns(["chosen","rejected"]).shuffle(seed=42).select(range(int(self.num_samples*0.1)))

        
#         # 转换为 DatasetDict 格式
#         filtered_dataset = DatasetDict({
#             "train": train_dataset["train"],
#             "eval":train_dataset["test"],   
#             "test": test_dataset
#         })
#         return filtered_dataset
    
class AdvBenchDataset(Dataset):
    think_model: bool = True
    def __init__(self,num_samples,think_mode) -> None:
        self.num_samples = num_samples
        self.think_mode = think_mode
    def get_dataset(self):
        template = user_cot_prompt_template if self.think_mode else user_prompt_template
        dataset = load_dataset("walledai/AdvBench")["train"]
        columns = dataset.column_names
        rej_answer = random.choice(reject_templates)
        # cot_answer = random.choice(cot_templates)
        
        self.dataset = dataset.map(lambda row:{
            "question":template.format(A=row["prompt"]),
            "label" : cot_answer if self.think_mode else rej_answer
        }).remove_columns(columns)
        
        self.dataset = self.dataset.train_test_split(test_size=0.1)
        return self.dataset    

        
        
        
class HExDataset(Dataset):
    think_model: bool = True
    def __init__(self,num_samples,think_mode) -> None:
        self.num_samples = num_samples
        self.think_mode = think_mode
        self.dataset = load_from_disk("/mnt/petrelfs/majiachen/datasets/HEx-PHI/cot_data")
        self.template = user_cot_prompt_template if self.think_mode else user_prompt_template
    def get_dataset(self):
        rej_answer = random.choice(reject_templates)
        self.dataset = self.dataset.map(lambda row:{
            "question":self.template.format(A=row["prompt"]),
            "label" : row["label"] if self.think_mode else rej_answer
        })
        self.dataset = self.dataset.train_test_split(test_size=0.1)
        return self.dataset        
        

class DoNotAnswerDataset(Dataset):
    think_model: bool = True
    def __init__(self,num_samples,think_mode) -> None:
        self.num_samples = num_samples
        self.think_mode = think_mode
        self.dataset = load_from_disk("/mnt/petrelfs/majiachen/datasets/DO-NOT-Answer")
        self.template = user_cot_prompt_template if self.think_mode else user_prompt_template
        
    def get_dataset(self):
        rej_answer = random.choice(reject_templates)
        self.dataset = self.dataset.map(lambda row:{
            "question":self.template.format(A=row["question"]),
            "label" : row["label"] if self.think_mode else rej_answer
        })
        self.dataset = self.dataset.train_test_split(test_size=0.1)
        return self.dataset        
        
               
        
        
        
        
        
class GeneralDataset(Dataset):
    think_mode: bool  = True
    def __init__(self,num_samples,think_mode) -> None:
        # self.subset = subset
        self.num_samples = num_samples
        self.think_mode = think_mode
        self.dataset = load_dataset("HuggingFaceH4/ultrachat_200k")["train_sft"]
    def get_dataset(self):
        columns = self.dataset.column_names
        template = user_cot_prompt_template if self.think_mode else user_prompt_template
        dataset = self.dataset.map(lambda row: {
            "question": template.format(A=row["messages"][0]["content"]),
            "label": row["messages"][1]["content"]        
        }).remove_columns(columns).shuffle(seed=42)
        self.dataset = dataset.filter(lambda row: len(row["question"]) < 300).select(range(self.num_samples))
        
        self.dataset = self.dataset.train_test_split(test_size=0.1)
        return self.dataset    
    
 
class XSTest(Dataset):
    '''
    XSTest comprises:
        -250 safe prompts across ten prompt types that well-calibrated models should not refuse to comply with.
        -200 unsafe prompts as contrasts that, for most LLM applications, should be refused.
    '''
    think_model: bool = True
    def __init__(self,think_mode) -> None:
        self.dataset = load_dataset("walledai/XSTest")
        self.think_mode = think_mode
        self.template = user_cot_prompt_template if self.think_mode else user_prompt_template
    def get_dataset(self):
        self.dataset = self.dataset.map(
            lambda row: {
                "question": self.template.format(A=row["prompt"]),
                "label": row["label"]
            })
        return self.dataset
            

class StrongREJECT(Dataset):
    '''
    A novel benchmark of 313 malicious prompts for use in evaluating jailbreaking attacks against LLMs, aimed to expose whether a jailbreak attack actually enables malicious actors to utilize LLMs for harmful tasks.
    '''
    think_model: bool = True
    def __init__(self,think_mode) -> None:
        self.dataset = load_dataset("walledai/StrongREJECT")
        self.think_mode = think_mode
        self.template = user_cot_prompt_template if self.think_mode else user_prompt_template
    def get_dataset(self):
        self.dataset = self.dataset.map(
            lambda row: {
                "question": self.template.format(A=row["prompt"]),
                "label": "unsafe"
            })
        return self.dataset
   

if __name__ == "__main__":
    prompt_config_path = 'config/cot_prompt_llm.yaml'
    task_configs = omegaconf.OmegaConf.load(prompt_config_path)
    # xstest = load_dataset("walledai/XSTest")
    
    # cot_data1 = DoNotAnswerDataset(num_samples=1200,think_mode=True)
    # ds1 = cot_data1.get_dataset()
    # cot_data2 = HExDataset(num_samples=1200,think_mode=True)
    # ds2 = cot_data2.get_dataset()
    
    # advbench = AdvBenchDataset(num_samples=1200,think_mode=True)
    # ds_adv = advbench.get_dataset()
    
    # general_ds = GeneralDataset(num_samples=1200,think_mode=True)
    # ds_gen = general_ds.get_dataset()
    # xstest = XSTest(task_configs=task_configs.mm_safetybench)
    # reject_dataset = xstest.get_dataset()
    # breakpoint()
    # reject_cot_dataset = xstest.get_cot_reject_dataset()



#### HEx-PHI数据集数据工程
    file_to_sub_label = {
        "category_1.csv": "Illegal Activity",
        "category_3.csv": "Hate,harass and violence",
        "category_4.csv": "Malware",
        "category_5.csv": "Physical Harm",
        "category_6.csv": "Economic Harm",
        "category_8.csv": "Fraud Deception",
        "category_7.csv": "Adult Content",
        "category_9.csv": "Political Campaigning",
        "category_10.csv": "Privacy Violation Activity",
        "category_11.csv": "Tailored Financial Advice"
    }
    def load_dataset(dataset_dir: str, task_configs: dict):
        dataset_entries = []
        for filename in os.listdir(dataset_dir):
            if filename.endswith(".csv"):
                sub_label = file_to_sub_label.get(filename)
                if not sub_label:
                    continue  # 如果文件不在字典中，跳过
                label_list = task_configs["HEx_PHI"].get(sub_label)
                if not label_list:
                    continue  # 如果该sub_label没有对应的label列表，跳过
                file_path = os.path.join(dataset_dir, filename)
                df = pd.read_csv(file_path, header=None)  # 读取文件，没有列名
                
                for _, row in df.iterrows():
                    label = random.choice(label_list) if label_list else None
                    dataset_entries.append({
                        "prompt": row[0],  # CSV中的每条数据
                        "harm_type": sub_label,
                        "label": label+random.choice(reject_cot_responses)
                    })
        return Dataset.from_pandas(pd.DataFrame(dataset_entries))
    dataset_dir = '/mnt/petrelfs/majiachen/datasets/HEx-PHI/csv'
    dataset = load_dataset(dataset_dir, task_configs)
    dataset.save_to_disk("/mnt/petrelfs/majiachen/datasets/HEx-PHI/cot_data")




# # #### Do not answer 数据集数据工程 (722data)
#     import yaml
#     dataset = load_dataset("LibrAI/do-not-answer")
#     with open('harm_types_output.yaml', 'r') as f:
#         harm_dict = yaml.safe_load(f)
#     new_data = []
#     for entry in dataset['train']:  # Assuming dataset is for the 'train' split
#         harm_type = entry.get('types_of_harm')
#         specific_harm = entry.get('specific_harms')
#         question = entry.get('question')
#         if harm_type is None or specific_harm is None:
#             print(f"Skipping entry due to missing fields: {entry}")
#             continue
#         if harm_type in harm_dict and specific_harm in harm_dict[harm_type]:
#             label = random.choice(harm_dict[harm_type][specific_harm])  # Randomly choose a response
#             harm_type = harm_type  # harm_type as label
#             specific_harm_type = specific_harm  # specific_harm as sub_label

#             new_data.append({
#                 "question": question,
#                 "label": label+random.choice(reject_cot_responses),
#                 "harm_type": harm_type,
#                 "specific_harm_type":specific_harm_type
#             })
#         else:
#             print(f"Skipping entry due to missing harm_type or specific_harm mapping: {entry}")

#     new_dataset = Dataset.from_dict({
#         'question': [item['question'] for item in new_data],
#         'label': [item['label'] for item in new_data],
#         'harm_type': [item['harm_type'] for item in new_data],
#         'specific_harm_type': [item['specific_harm_type'] for item in new_data]})
    
#     new_dataset.save_to_disk("/mnt/petrelfs/majiachen/datasets/DO-NOT-Answer")
#     with open("/mnt/petrelfs/majiachen/datasets/DO-NOT-Answer/harm_cot_data_output.json", "w") as output_file:
#         json.dump(new_data, output_file, indent=4)
#     print("Processing complete. The results have been saved as a dataset and to 'harm_data_output.json'.")
