
ngpu=8
ncpu=16
WANDB_PROJECT=llm-cot-safety
config_file=config/deepspeed_zero2.yaml


model_name=SmolVLM
think_mode=True

version=v1_llama
train_task_names=mmsafetybench+sharedgpt4v_${version}
base_dir=/mnt/lustrenew/mllm_safety-shared/tmp/majiachen/results/model:sft_mllm_${model_name}/train:${train_task_names}


echo "training..."
echo "run_name1 : $base_dir"
mkdir -p $base_dir


WANDB_PROJECT=${WANDB_PROJECT} PYTHONPATH=. srun -p mllm_safety --quotatype=spot --gres=gpu:${ngpu} --cpus-per-task=${ncpu} --time=30000 \
    accelerate launch --config_file ${config_file} --num_processes ${ngpu} --main_process_port $(( RANDOM % 1000 + 30000 )) \
    src/sft_vlm_cot_.py \
    --dataset_name aa \
    --output_dir ${base_dir} \
    --model_identifier ${model_name} \
    --version ${version} \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_steps 10 \
    --logging_strategy "steps" \
    --save_steps 200 \

    # --use_peft \
    # --lora_target_modules down_proj, o_proj, k_proj, q_proj, gate_proj, up_proj, v_proj

