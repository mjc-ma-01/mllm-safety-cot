test_dataset="ood"
num_gpus=4 # 使用的 GPU 数量
world_size=4  # 任务总数量/数据集切分块总数
batch_size=4

model_name=llava1.5_7b # 选模型
use_peft=False
think_mode=True 



# 声明一个关联数组来模拟字典
declare -A model_map
# 给关联数组赋值
model_map["pixtral_12b"]="/mnt/lustrenew/mllm_safety-shared/models/huggingface/mistralai/Pixtral-12B-2409"
model_map["gamma3_4b"]="/mnt/lustrenew/mllm_safety-shared/models/huggingface/google/gemma-3-4b-it"
model_map["SmolVLM"]="/mnt/lustrenew/mllm_safety-shared/models/huggingface/HuggingFaceTB/SmolVLM-Instruct"
model_map["llava1.5_7b"]="/mnt/lustrenew/mllm_safety-shared/models/huggingface/llava-hf/llava-v1.6-mistral-7b-hf"
model_map["qwen_7b_vl"]="/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2-VL-7B-Instruct"
model_map["llama_11b"]="/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct"

version=origin
train_task_names=${version}
path=${model_map["$model_name"]}


# for rank in $(seq 0 $((world_size - 1))); do
    
#     save_path=./logs/sft_answer/model:sft_mllm_${model_name}/train:${train_task_names}/test:${test_dataset}/$(printf "%05d" ${rank})-$(printf "%05d" ${world_size}).json
    
#     PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=4 --time=30000 \
#      python src/inference_mllm.py \
#     --model_identifier ${model_name} \
#     --use_peft ${use_peft} \
#     --model_path ${path} \
#     --save_log_path  ${save_path} \
#     --test_dataset ${test_dataset} \
#     --cot ${think_mode} \
#     --world_size ${world_size} \
#     --rank ${rank} \
#     --batch_size ${batch_size} &
# done
# wait  



############################## 用score模型给判断是否安全（危险问题拒答、非危险问题做出回答） ##############################
scores=()  # 存放所有得分的数组
output_dir="./logs/sft_answer/model:sft_mllm_${model_name}/train:${train_task_names}/test:${test_dataset}"  # 结果存放目录
mkdir -p "${output_dir}"
tmp_score_file="${output_dir}/tmp_scores.txt"
> "$tmp_score_file"

for rank in $(seq 0 $((world_size - 1))); do
    gpu_id=$((rank % num_gpus))  # 根据 rank 分配 GPU
    file="${output_dir}/$(printf "%05d" ${rank})-$(printf "%05d" ${world_size}).json"
    (
        score=$(PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=4 --time=30000 \
        python src/eval_mllm.py \
            --model_path "/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct" \
            --input_path ${file} \
            --save_score_path ${file} \
            --batch_size ${batch_size} | tail -n 1)
        if [[ "$score" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "$score" >> "$tmp_score_file"  # 仅写入数值
            echo "Rank ${rank} score: ${score}" 
        else
            echo "Rank ${rank} returned an invalid score: ${score}"
        fi
    ) &
done
wait  

################################ 计算平均得分 ####################################
total=0
count=0
while IFS= read -r s; do
    if [[ "$s" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then  # 确保是数值
        total=$(awk "BEGIN {print $total + $s}")  
        count=$((count + 1))
    fi
done < "$tmp_score_file"
if [[ "$count" -gt 0 ]]; then
    avg_score=$(awk "BEGIN {print $total / $count}")else
    avg_score="N/A (No valid scores)"
fi
final_score_file="${output_dir}/final_scores.txt"
echo "Final average score: ${avg_score}" | tee -a "$final_score_file"
rm -f "$tmp_score_file"
echo "All scores and final average score saved to ${final_score_file}"