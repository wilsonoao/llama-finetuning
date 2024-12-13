#!/bin/bash

#SBATCH -c 10
#SBATCH -t 0-4:00
#SBATCH --mem=96G
#SBATCH -p gpu_dia
#SBATCH --gres=gpu:1
#SBATCH -o ./jobs_log/%j.out
#SBATCH -e ./jobs_log/%j.out


module load dgx
module load cuda/12.1
module load miniconda3/24.1.2

source activate llama3.2

main_folder="/n/scratch/users/f/fas994/wilson/fine-tune/train_800_img_rank_10_rank_16"
data_path="/n/scratch/users/f/fas994/wilson/data_csv/train_800_img_rank_10/test.csv"

processed_folders=()

while true; do
	new_folder_found=false

	for subfolder in "$main_folder"/*/; do

		if [[ ! " ${processed_folders[@]} " =~ " ${subfolder} " ]]; then

			processed_folders+=("$subfolder")
			if [ -f "${subfolder}test.csv" ]  || [ ! -f "${subfolder}adapter_model.safetensors" ]; then
				continue
			fi

			srun  	python tumor_infer.py \
				--data_file_path "${data_path}" \
				--temperature 0.5 \
				--top_p 0.8 \
				--peft_model_path $subfolder \
				--output_csv "${subfolder}test.csv" 
			
			new_folder_found=true
		fi
	done

	if [ "$new_folder_found" = false ]; then
		echo "No new folders found. Exiting."
		srun    python  /n/scratch/users/f/fas994/wilson/fine-tune/metric.py \
			--input_csv "${main_folder}"
		break
	fi

done
	
