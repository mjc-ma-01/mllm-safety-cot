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
accelerate launch \
    --config_file config/deepspeed_zero2.yaml \
    --num_processes 8 \
    src/sft_vlm.py \
    --dataset_name aa \
    --model_name_or_path /fs-computility/ai-shen/mllm_safety-shared/models/huggingface/meta-llama/Llama-3.2-11B-Vision-Instruct \
    --per_device_train_batch_size 1 \
    --save_strategy steps \
    --save_steps 100 \
    --gradient_accumulation_steps 1 \
    --output_dir /fs-computility/ai-shen/majiachen/results/sft_llama+sharegpt4v01 \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_strategy="steps" \
    --logging_steps 1 \
    --save_only_model \

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
#     --logging_steps 1