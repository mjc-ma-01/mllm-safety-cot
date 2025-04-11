from datasets import load_dataset
import json,os,shutil
from huggingface_hub import hf_hub_download,list_repo_files
# repo_id = "kzhou35/mssbench"
# sub_folder = "chat/"
# save_dir = "/fs-computility/ai-shen/majiachen/datasets/MSSbench"
# os.makedirs(save_dir,exist_ok=True,)
# files = list_repo_files(repo_id = repo_id,repo_type="dataset")
# files_ = [f for f in files if f.startswith(sub_folder)]
# for f in files_:
#     print(f"正在下载{f}")
#     file_path = hf_hub_download(repo_id=repo_id,filename = f, repo_type="dataset")
#     local_file_path = os.path.join(save_dir,"chat", os.path.basename(f))
#     shutil.copy(file_path, local_file_path)

# breakpoint()

# from datasets import load_dataset

# ds = load_dataset("kzhou35/mssbench")
# breakpoint()


#"lmsys/vicuna-7b-v1.5"
from huggingface_hub import snapshot_download

# Specify the repository ID
# repo_id = "google/gemma-2-9b"
# repo_id = "HuggingFaceH4/zephyr-7b-beta"
# repo_id = "allenai/OLMo-7B-0424-hf"
# repo_id = "lmsys/vicuna-7b-v1.5"
# repo_id = "meta-llama/Llama-Guard-3-8B"
# repo_id = "sinwang/SIUO"
# repo_id = "kzhou35/mssbench"


# repo_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# refer to : https://huggingface.co/deepseek-ai/deepseek-vl2-small
# repo_id = "deepseek-ai/deepseek-vl2-small"

## refer to : https://huggingface.co/OpenGVLab/InternVL2_5-8B  ## Finetune on https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html
# repo_id = "OpenGVLab/InternVL2_5-8B" 

# repo_id = "llava-hf/llava-v1.6-mistral-7b-hf"
# repo_id = "mistralai/Pixtral-12B-2409"
# repo_id = "allenai/Molmo-7B-D-0924"
# repo_id = "Qwen/Qwen2-VL-7B-Instruct"

# repo_id = "HuggingFaceTB/SmolVLM-Instruct"
repo_id = "google/gemma-3-4b-it"





revision = None
local_directory = f"/mnt/lustrenew/mllm_safety-shared/models/huggingface/{repo_id}"
# local_directory = f"/mnt/lustrenew/mllm_safety-shared/datasets/SIUO"

# Download the repository files
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    revision=revision,
    local_dir=local_directory,
    local_dir_use_symlinks=False,  # Avoid symlinks for easier file handling
)

print(f"Files downloaded to: {local_directory}")