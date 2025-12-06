

model_path=${1:-Qwen/Qwen2.5-7B-Instruct}   
path_after_slash=$(basename "$model_path") 
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory


CUDA_VISIBLE_DEVICES=1 python safegrad_train.py \
	--model_name_or_path ${model_path} \
	--data_path PKU-Alignment/BeaverTails_safe \
	--bf16 True \
	--output_dir ckpt/${path_after_slash}_sft \
	--num_train_epochs 10 \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--gradient_accumulation_steps 1 \
	--eval_strategy "no" \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  1e-3 \
	--weight_decay 0 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "cosine" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer "normal" \

cd poison/evaluation  

CUDA_VISIBLE_DEVICES=1 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_sft \
	--model_folder ${model_path} \
	--output_path ../../data/poison/${path_after_slash}_sft

CUDA_VISIBLE_DEVICES=1 python eval_sentiment.py \
	--input_path ../../data/poison/${path_after_slash}_sft