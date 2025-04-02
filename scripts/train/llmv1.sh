ngpu=8
ncpu=16
WANDB_PROJECT=llm-cot-safety
config_file=config/deepspeed_zero2.yaml

model_name=sft_llm_zephyer
version=v1
train_task_names=AdvBench_520+ultrachat_1200_${version}

# model_path=/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct
# model_path=/mnt/lustrenew/mllm_safety-shared/models/huggingface/lmsys/vicuna-7b-v1.5

model_path=/mnt/lustrenew/mllm_safety-shared/models/huggingface/HuggingFaceH4/zephyr-7b-beta

base_dir=/mnt/lustrenew/mllm_safety-shared/tmp/majiachen/results/model:${model_name}/train:${train_task_names}

think_mode=True
num_train_epochs=12
per_device_train_batch_size=4
gradient_accumulation_steps=$((16 / ngpu))
per_device_eval_batch_size=4
learning_rate=1e-4
eval_steps=0.25
save_steps=${eval_steps}
save_only_model=False
save_total_limit=5
load_best_model_at_end=True


echo "training..."
echo "run_name: $base_dir"
mkdir -p $base_dir

WANDB_PROJECT=${WANDB_PROJECT} PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:${ngpu} --cpus-per-task=${ncpu} --time=30000 \
    accelerate launch --config_file ${config_file} --num_processes ${ngpu} --main_process_port $(( RANDOM % 1000 + 30000 )) \
    src/sft_llm_cot.py \
    --model_path ${model_path} \
    --output_dir ${base_dir} \
    --run_name ${base_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --learning_rate ${learning_rate} \
    --eval_steps ${eval_steps} \
    --save_steps ${save_steps} \
    --save_only_model ${save_only_model} \
    --save_total_limit ${save_total_limit} \
    --load_best_model_at_end ${load_best_model_at_end}

