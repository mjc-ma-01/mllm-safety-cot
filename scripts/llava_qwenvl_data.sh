
ngpu=8
ncpu=16
WANDB_PROJECT=llm-cot-safety
config_file=config/deepspeed_zero2.yaml

model_name=llava1.5_7b

version=v2_qwenvl
think_mode=True
train_task_names=mmsafetybench+sharedgpt4v_${version}
base_dir=/mnt/lustrenew/mllm_safety-shared/tmp/majiachen/results/model:sft_mllm_${model_name}/train:${train_task_names}

echo "training..."
echo "run_name: $base_dir"
mkdir -p $base_dir

WANDB_PROJECT=${WANDB_PROJECT} PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:${ngpu} --cpus-per-task=${ncpu} --time=30000 \
    accelerate launch --config_file ${config_file} --num_processes ${ngpu} --main_process_port $(( RANDOM % 1000 + 30000 )) \
    src/sft_vlm_cot_.py \
    --dataset_name aa \
    --version ${version} \
    --model_identifier ${model_name} \
    --output_dir ${base_dir} \
    --save_steps 200 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_steps 10 \
    --logging_strategy "steps" \
    --gradient_checkpointing \

    # --use_peft  \

    # --lora_alpha 16 \
    # --lora_taget_modules q_proj, v_proj 

    # --dataset_name aa \
    # --model_name_or_path ${model_path} \
    # --output_dir ${base_dir} \
    # --use_peft True \
    # --per_device_train_batch_size=1 \
    # --gradient_accumulation_steps=1 \
    # --save_strategy steps \
    # --save_steps 100 \
    # --bf16 \
    # --torch_dtype bfloat16 \
    # --gradient_checkpointing \
    # --logging_steps 1 \
    # --logging_strategy "steps" \
    # --save_only_model \
    # --trust_remote_code True \



# srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=30000 accelerate launch \
#     --config_file config/deepspeed_zero2.yaml \
#     sft_vlm.py \
#     --dataset_name aa \
#     --model_name_or_path /mnt/hwfile/llm-safety/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --output_dir /mnt/hwfile/llm-safety/models/_tmp/mjc/sft_llama \
#     --bf16 \
#     --torch_dtype bfloat16 \
#     --logging_strategy="steps" \
#     --logging_steps 1 \
#     --save_only_model


