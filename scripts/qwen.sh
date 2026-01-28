ngpu=8
ncpu=16
WANDB_PROJECT=llm-cot-safety
config_file=config/deepspeed_zero2.yaml

model_name=qwen_7b_vl

version=v2
think_mode=True

train_task_names=mmsafetybench+sharedgpt4v_${version}
base_dir=/results/model:sft_mllm_${model_name}/train:${train_task_names}


echo "training..."
echo "run_name: $base_dir"
mkdir -p $base_dir

WANDB_PROJECT=${WANDB_PROJECT} PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:${ngpu} --cpus-per-task=${ncpu} --time=30000 \
    accelerate launch --config_file ${config_file} --num_processes ${ngpu} --main_process_port $(( RANDOM % 1000 + 30000 )) \
    src/sft_vlm_cot_.py \
    --dataset_name aa \
    --model_identifier ${model_name} \
    --version ${version} \
    --output_dir ${base_dir} \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --save_steps 200 \
    --gradient_checkpointing \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_steps 10 \
    --learning_rate 1e-4 \
    --logging_strategy "steps" \
    --use_peft \
    --lora_alpha 16 \
    --lora_r 64 \
    --lora_target_modules q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj \
    # --lora_dropout 0.05 \
    # --lora_task_type CAUSAL_LM \
# think_mode=True
# per_device_train_batch_size=4
# gradient_accumulation_steps=$((16 / ngpu))
# per_device_eval_batch_size=4
# learning_rate=2e-5
# save_total_limit=5
# load_best_model_at_end=True


# config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     inference_mode=False,  # 训练模式
#     r=64,  # Lora 秩
#     lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
#     lora_dropout=0.05,  # Dropout 比例
#     bias="none",
# )
