from datasets import Dataset, DatasetDict
from typing import Optional, Sequence, List, Dict
import os
from datasets import load_dataset, concatenate_datasets
import json, random
import transformers
from pathlib import Path
import omegaconf
from tqdm import tqdm
# srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 python dataset/base.py
reject_templates = [
    "I'm sorry, but I am unable to assist with that request.",
    "My apologies, I can't comply with this request.",
    "Sorry, I can't fulfill your request at the moment.",
    "I'm sorry, I cannot help with this request.",
    "My apologies, I am unable to process that request.",
    "Sorry, I'm not able to comply with this request.",
    "I'm sorry, I can't provide that information.",
    "My apologies, I cannot provide assistance with this.",
    "Sorry, but I cannot fulfill this request.",
    "I'm sorry, but I'm unable to help with this request."
]

cot_templates = [
    "It seems the user is asking for content related to unlawful actions, which goes against our safety protocols. Therefore, I cannot provide an answer to this inquiry.",
    "The question contains references to harmful or illegal activities, violating our guidelines. Consequently, I am unable to assist with this request.",
    "This request appears to involve sensitive or unsafe content that conflicts with our safety guidelines. As such, I cannot fulfill this request.",
    "The user's question raises concerns about potentially dangerous or inappropriate material. In accordance with our safety policies, I am unable to generate a response.",
    "It looks like the user is seeking information related to harmful behavior, which violates our safety rules. For that reason, I cannot respond to this query.",
    "This query seems to involve illegal or unsafe topics, which are not allowed under our safety guidelines. Therefore, I am unable to provide assistance.",
    "The content in the user's question touches on potentially dangerous subjects that breach our safety standards. As a result, I am unable to process this request.",
    "It appears that the question is related to unlawful activities or harmful behaviors. This violates our security protocols, and I cannot comply with such a request.",
    "The user's inquiry suggests a request for content linked to dangerous or harmful topics. Due to our safety policies, I am unable to assist with this.",
    "The question seems to involve potentially illegal or unsafe material. In line with our safety guidelines, I cannot proceed with answering this.",
    "The user's request involves topics that are prohibited due to their harmful nature. As this violates our safety guidelines, I cannot fulfill this request.",
    "It seems the user is asking about topics that are in violation of our safety policies, such as harmful or illegal activities. As such, I must decline to provide a response.",
    "This request contains references to harmful or illegal behavior, which is against our safety standards. For that reason, I cannot offer any help with this.",
    "The question appears to touch on sensitive, potentially illegal content. As a result, I am unable to assist, as it violates our safety protocols.",
    "It seems that this request involves content related to harmful or illegal activities, which is not allowed by our guidelines. Therefore, I am unable to fulfill it.",
    "The user's question contains content that is not in compliance with our safety guidelines due to its potential harm. For this reason, I cannot provide an answer.",
    "The request appears to contain references to dangerous or illegal actions. Given our safety guidelines, I cannot assist with such inquiries.",
    "It seems the user is asking for information related to unsafe or illegal topics. This violates our safety policies, and I am unable to comply with such a request.",
    "This query seems to involve subjects that go against our safety guidelines. Due to the harmful nature of the content, I cannot fulfill this request.",
    "The user's request pertains to potentially harmful content that violates our safety regulations. Therefore, I must refrain from answering."
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
        cot_answer = random.choice(cot_templates)
        
        self.dataset = dataset.map(lambda row:{
            "question":template.format(A=row["prompt"]),
            "label" : cot_answer if self.think_mode else rej_answer
        }).remove_columns(columns)
        
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
        self.dataset = dataset.filter(lambda row: len(row["question"]) < 500).select(range(self.num_samples))
        
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
    # breakpoint()
    advbench = AdvBenchDataset(num_samples=1200,think_mode=True)
    ds_adv = advbench.get_dataset()
    
    general_ds = GeneralDataset(num_samples=1200,think_mode=True)
    ds_gen = general_ds.get_dataset()
    # xstest = XSTest(task_configs=task_configs.mm_safetybench)
    # reject_dataset = xstest.get_dataset()
    # breakpoint()
    # reject_cot_dataset = xstest.get_cot_reject_dataset()
