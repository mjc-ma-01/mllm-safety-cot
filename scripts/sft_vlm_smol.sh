
ngpu=8
ncpu=16
WANDB_PROJECT=llm-cot-safety
config_file=config/deepspeed_zero2.yaml


model_name=sft_mllm_SmolVLM

version=v1
train_task_names=mmsafetybench+sharedgpt4v_${version}
base_dir=/mnt/lustrenew/mllm_safety-shared/tmp/majiachen/results/model:${model_name}/train:${train_task_names}


model_path=/mnt/lustrenew/mllm_safety-shared/models/huggingface/HuggingFaceTB/SmolVLM-Instruct

echo "training..."
echo "run_name: $base_dir"
mkdir -p $base_dir


WANDB_PROJECT=${WANDB_PROJECT} PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:${ngpu} --cpus-per-task=${ncpu} --time=30000 \
    accelerate launch --config_file ${config_file} --num_processes ${ngpu} --main_process_port $(( RANDOM % 1000 + 30000 )) \
    src/sft_vlm_smol.py \
    --dataset_name aa \
    --model_name_or_path ${model_path} \
    --output_dir ${base_dir} \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --bf16 \
    --num_train_epochs 2 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --logging_steps 1 \
    --logging_strategy "steps" \
    --use_peft \
    --lora_target_modules down_proj, o_proj, k_proj, q_proj, gate_proj, up_proj, v_proj

