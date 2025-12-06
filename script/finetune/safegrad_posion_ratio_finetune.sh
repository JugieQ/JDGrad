#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate SafeGrad

poison_ratio=${1:-0.1}
device=${2:-2}
sample_num=${3:-1000}
model_path=${4:-meta-llama/Meta-Llama-3-8B-Instruct}
guide_data_num=${5:-100}
RHO=${6:-1}
dataset_name=${7:-gsm8k}
benign_dataset=${8:-data/gsm8k.json}  

path_after_slash=$(basename "$model_path")
output_dir=ckpt/${dataset_name}/${path_after_slash}_safegrad_f_${RHO}_${poison_ratio}_${sample_num}_${guide_data_num}
poison_eval_out=data/poison/${dataset_name}/${path_after_slash}_safegrad_f_${RHO}_${poison_ratio}_${sample_num}_${guide_data_num}
acc_eval_out=data/${dataset_name}/${path_after_slash}_safegrad_f_${RHO}_${poison_ratio}_${sample_num}_${guide_data_num}

mkdir -p "$(dirname "$output_dir")"
mkdir -p "$(dirname "$poison_eval_out")"
mkdir -p "$(dirname "$acc_eval_out")"

echo "======================= RUN CONFIG ======================="
echo "[Device ID]             : $device"
echo "[RHO]                   : $RHO"
echo "[Poison Ratio]          : $poison_ratio"
echo "[Sample Number]         : $sample_num"
echo "[Guide Data Num]        : $guide_data_num"
echo "[Dataset Name]          : $dataset_name"
echo "[Benign Dataset]        : $benign_dataset"
echo "[Model Path]            : $model_path"
echo "[Output Directory]      : $output_dir"
echo "=========================================================="

cd ../../  

# ---------------- train task ----------------
echo "Starting training..."
CUDA_VISIBLE_DEVICES=${device} python safegrad_train.py \
  --model_name_or_path ${model_path} \
  --data_path PKU-Alignment/BeaverTails_dangerous \
  --bf16 True \
  --output_dir  ${output_dir} \
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
  --optimizer safegrad \
  --eval_strategy "steps" \
  --sample_num ${sample_num} \
  --poison_ratio ${poison_ratio} \
  --label_smoothing_factor 0 \
  --benign_dataset ${benign_dataset} \
  --rho ${RHO} \
  --guide_data_num ${guide_data_num}

# ---------------- posion evaluation ----------------
cd poison/evaluation
echo "Evaluating backdoor behavior..."

CUDA_VISIBLE_DEVICES=${device} python finetune_defense_pred.py \
  --lora_folder ../../${output_dir} \
  --model_folder ${model_path} \
  --output_path ../../${poison_eval_out}

CUDA_VISIBLE_DEVICES=${device} python eval_sentiment_llamaguard.py \
  --input_path ../../${poison_eval_out}

# ---------------- fintune accuracy evaluation ----------------
cd ../../${dataset_name}
echo "Evaluating accuracy on benign dataset..."

CUDA_VISIBLE_DEVICES=${device} python finetune_defense_pred_eval.py \
  --lora_folder ../${output_dir} \
  --model_folder ${model_path} \
  --output_path ../${acc_eval_out}

