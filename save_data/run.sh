#!/bin/bash

#SBATCH -c 10
#SBATCH -t 0-24:00
#SBATCH --mem=96G
#SBATCH -p gpu_dia
#SBATCH --gres=gpu:1
#SBATCH -o ./jobs_log/%j.out
#SBATCH -e ./jobs_log/%j.out


module load dgx
module load cuda/12.1
module load miniconda3/24.1.2

source activate llama3.2


finetune_dir="More_script_train_800_img_rank_10"
disciption="_32"
dataset_path="/n/scratch/users/f/fas994/wilson/data_csv/${finetune_dir}/train.csv"


out_dir="$finetune_dir$disciption"
start_run=1
for folder in "/n/scratch/users/f/fas994/wilson/fine-tune/$out_dir"/epoch_*;do
	if [ -d "$folder" ]; then
		number=$(echo "$folder" | grep -oP 'epoch_\K\d+')

		if [ "$number" -gt "$start_run" ]; then
			start_run=$number
		fi
	fi
done

find_free_port() {
	while true; do
		PORT=$(python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
		if ! lsof -i:$PORT > /dev/null; then
			echo $PORT
			break
		fi
	done
}

MASTER_PORT=$(find_free_port)

out_dir="$finetune_dir$disciption"
num_runs=$((start_run + 30))
for i in $(seq $start_run $num_runs)
do	
	output_dir_epoch=$i
	peft_check_point_epoch=$((i - 1))
	from_check_point_path=/n/scratch/users/f/fas994/wilson/fine-tune/${out_dir}/epoch_$peft_check_point_epoch

	if [ "$i" -eq 1 ]; then
		from_check_point_path=""
	fi

	srun	torchrun --nnodes 1 --nproc_per_node 1 --master_port=$MASTER_PORT /n/scratch/users/f/fas994/wilson/llama32/llama-recipes/recipes/quickstart/finetuning/finetuning.py \
		--enable_fsdp \
		--lr 5e-5 \
		--num_epochs 2 \
		--batch_size_training 2 \
		--model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
		--dist_checkpoint_root_folder "/n/scratch/users/f/fas994/wilson/finetuning/" \
		--dist_checkpoint_folder fine-tuned \
		--use_fast_kernels \
		--gradient_accumulation_steps 16 \
		--dataset "custom_dataset" \
		--custom_dataset.test_split "test" \
		--custom_dataset.file "/n/scratch/users/f/fas994/wilson/llama32/llama-recipes/recipes/quickstart/finetuning/datasets/tumor_dataset.py" \
		--custom_dataset.data_path "${dataset_path}" \
		--run_validation True \
		--batching_strategy padding \
		--use_peft True \
		--from_peft_checkpoint "$from_check_point_path" \
		--peft_method lora \
		--lora_config.r  8 \
		--lora_config.lora_alpha 32 \
		--use_wandb True \
		--project "${out_dir}" \
		--output_dir "/n/scratch/users/f/fas994/wilson/fine-tune/"${out_dir}"/epoch_$output_dir_epoch" \
		--gamma 0.95 
done


