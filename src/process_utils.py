import json
from datasets import Dataset

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

# def extract_question(label):
#     import re
#     match = re.search(r"(?:is:|question is:|Question is:|question:|would be:|could be:)\s*(.*)", label, re.IGNORECASE)
#     if match:
#         result = match.group(1).strip()
#         # 去除拒绝模板中的任何句子
#         for template in rejection_templates:
#             result = result.replace(template, "").strip()
#         return result
#     return None
# with open("./logs/cot_training_data/cot_data_useful.json", "r") as f:
#     data = json.load(f)
# dataset = Dataset.from_list(data)
# ds = dataset.map(
#     lambda row: {"question_useful": extract_question(row["label"])},
# ).filter(lambda row: row["question_useful"] is not None)
# ds_ = ds.to_list()
# save_log_path = f"./logs/cot_training_data/cot_data_useful_.json"
# with open(save_log_path, "w") as file:
#     json.dump(ds_, file, indent=4)
    
# # dataset.save_to_disk("./dataset/cot_data_useful")


## process new data
# 读取原始数据集
with open("./logs/cot_training_data/cot_data_useful.json", "r") as f:
    data = json.load(f)

# 使用datasets库加载数据集
ds = Dataset.from_list(data)
# 定义模板列表
templatelist = [
    "I cannot confidently",
    "I cannot assist",
    "I cannot provide",
    "I cannot identify",
    "I cannot help",
    "I can't help",
    "I'm sorry",
    "I'm not able",
    "I'm unable to",
    "I'm not capable",
    "I am unable to",
    "I cannot"
]

# 定义过滤条件
def filter_labels(example):
    label = example.get('label', '').lower()
    # 如果label字段包含templatelist中的任意字符串，返回False表示要移除
    return not any(template.lower() in label for template in templatelist)

# 应用过滤条件，返回新的数据集
filtered_dataset = ds.filter(filter_labels)
ds_ = filtered_dataset.to_list()
with open("./logs/cot_training_data/cot_data_useful_tmp.json", "w") as file:
    json.dump(ds_, file, indent=4)
    
# 查看过滤后的数据集
# print(filtered_dataset)
ds.save_to_disk("./dataset/cot_data_useful")
