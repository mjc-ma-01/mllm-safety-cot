ngpu=8
ncpu=16
WANDB_PROJECT=llm-cot-safety
config_file=config/deepspeed_zero2.yaml


# model_name=sft_mllm_qwenvl_7b
model_name=sft_mllm_qwenvl_7b_test

version=v1
train_task_names=mmsafetybench+sharedgpt4v_${version}
base_dir=/mnt/lustrenew/mllm_safety-shared/tmp/majiachen/results/model:${model_name}/train:${train_task_names}

# model_path=/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct
# model_path=/mnt/lustrenew/mllm_safety-shared/models/huggingface/llava-hf/llava-v1.6-mistral-7b-hf
model_path=/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2-VL-7B-Instruct

echo "training..."
echo "run_name: $base_dir"
mkdir -p $base_dir


WANDB_PROJECT=${WANDB_PROJECT} PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:${ngpu} --cpus-per-task=${ncpu} --time=30000 \
    accelerate launch --config_file ${config_file} --num_processes ${ngpu} --main_process_port $(( RANDOM % 1000 + 30000 )) \
    src/sft_vlm_cot.py \
    --dataset_name aa \
    --use_peft True \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --model_name_or_path ${model_path} \
    --save_strategy steps \
    --save_steps 100 \
    --output_dir ${base_dir} \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_steps 10 \
    --logging_strategy "steps" \
    --save_only_model \

    # --lora_dropout 0.05 \
    # --lora_task_type CAUSAL_LM \
# think_mode=True
# per_device_train_batch_size=4
# gradient_accumulation_steps=$((16 / ngpu))
# per_device_eval_batch_size=4
# learning_rate=2e-5
# save_total_limit=5
# load_best_model_at_end=True



    # --save_total_limit=5 \
    # --load_best_model_at_end=True
    # --save_only_model \

# accelerate launch \
#     --config_file config/deepspeed_zero2.yaml \
#     --num_processes 8 \
#     src/sft_vlm_cot.py \
#     --dataset_name aa \
#     --model_name_or_path /fs-computility/ai-shen/mllm_safety-shared/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct \
#     --per_device_train_batch_size 1 \
#     --save_strategy steps \
#     --save_steps 100 \
#     --gradient_accumulation_steps 1 \
#     --output_dir /fs-computility/ai-shen/majiachen/results/sft_cot_llama+sharegpt4v02 \
#     --bf16 \
#     --torch_dtype bfloat16 \
#     --logging_strategy="steps" \
#     --logging_steps 1 \
#     --save_only_model \

# 这个是对应的启动脚本，里面deepspeed的config路径可能要换一下~
# ngpu=8
# ncpu=16
# time=30000

# srun -p mllm-align --quotatype=reserved --gres=gpu:$ngpu --cpus-per-task=$ncpu --time=$time accelerate launch \
#     --config_file configs/accelerate_configs/deepspeed_zero2.yaml \
#     --num_processes $ngpu \
#     sft_vlm.py \
#     --dataset_name /mnt/hwfile/llm-safety/datasets/HuggingFaceH4___llava-instruct-mix-vsft \
#     --model_name_or_path /mnt/hwfile/llm-safety/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --output_dir ./model/ \
#     --bf16 \
#     --torch_dtype bfloat16 \
#     --logging_strategy="steps" \
#     --logging_steps 1=