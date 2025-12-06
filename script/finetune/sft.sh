#!/bin/bash

# ---------- env ----------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SafeGrad

poison_ratio=${1:-0.1}
device=${2:-2}
sample_num=1000
model_path=meta-llama/Meta-Llama-3-8B-Instruct
benign_dataset=data/sst2.json

path_after_slash=$(basename "$model_path")
output_dir=ckpt/sst2/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}
poison_eval_out=data/poison/sst2/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}
acc_eval_out=data/sst2/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}

mkdir -p "$(dirname "$output_dir")"
mkdir -p "$(dirname "$poison_eval_out")"
mkdir -p "$(dirname "$acc_eval_out")"

echo "======================= RUN CONFIG ======================="
echo "[RHO]                   : $RHO"
echo "[Poison Ratio]          : $poison_ratio"
echo "[Sample Number]         : $sample_num"
echo "[Benign Dataset]        : $benign_dataset"
echo "[Model Path]            : $model_path"
echo "[Output Directory]      : $output_dir"
echo "=========================================================="

# ---------- Train ----------
cd ../../
CUDA_VISIBLE_DEVICES=${device} python safegrad_train.py \
  --model_name_or_path "${model_path}" \
  --lora_folder ckpt/sst2/${path_after_slash}_sft_f_${poison_ratio}_${sample_num} \
  --data_path PKU-Alignment/BeaverTails_dangerous \
  --bf16 True \
  --output_dir "${output_dir}" \
  --num_train_epochs 10 \
  --per_device_train_batch_size 10 \
  --per_device_eval_batch_size 10 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 100000 \
  --save_total_limit 0 \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "constant" \
  --logging_steps 10 \
  --tf32 True \
  --eval_steps 5000 \
  --cache_dir cache \
  --optimizer sft \
  --evaluation_strategy "steps" \
  --sample_num "${sample_num}" \
  --poison_ratio "${poison_ratio}" \
  --label_smoothing_factor 0 \
  --benign_dataset "${benign_dataset}" \


# ---------- Poison Evaluation ----------
cd poison/evaluation
CUDA_VISIBLE_DEVICES=${device} python pred.py \
  --lora_folder ckpt/sst2/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}  \
  --model_folder "${model_path}" \
  --output_path ../../${poison_eval_out}

CUDA_VISIBLE_DEVICES=${device} python eval_sentiment.py \
  --input_path ../../${poison_eval_out}

# ---------- Accuracy Evaluation ----------
cd ../../sst2
CUDA_VISIBLE_DEVICES=${device} python pred_eval.py \
  --lora_folder ckpt/sst2/${path_after_slash}_sft_f_${poison_ratio}_${sample_num} \
  --model_folder "${model_path}" \
  --output_path ../${acc_eval_out}
