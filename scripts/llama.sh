
ngpu=8
ncpu=16
WANDB_PROJECT=llm-cot-safety
config_file=config/deepspeed_zero2.yaml


model_name=llama_11b
think_mode=True

version=v2_useful
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
    --think_mode ${think_mode} \
    --model_identifier ${model_name} \
    --output_dir ${base_dir} \
    --num_train_epochs 2 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_steps 10 \
    --logging_strategy "steps" \
    # --use_peft \
    # --lora_target_modules  down_proj, o_proj, k_proj, q_proj, gate_proj, up_proj, v_proj 

