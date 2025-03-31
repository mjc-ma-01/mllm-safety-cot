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