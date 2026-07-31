#!/bin/bash

#SBATCH -J test
#SBATCH -p gpu
#SBATCH -o sbatch_printout/filename_%j.txt
#SBATCH -e sbatch_printout/filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hy17@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00
#SBATCH -A r00105

module load conda
conda activate /N/slate/hy17/envs/ladiff
cd /N/slate/hy17/Projects/LaDiffCodec

python main.py --synthesis --load_model saved_models/1116_AE_1_5/model_best.amlt  --seq_len_p_sec 3.2 --inp_channels 64 --diff_dims 256 --continuous_type AE --discrete_type Encodec --cond_channels 128 --cond_bandwidth 1 --ratios 8 5 4 2 --input_dir /N/project/SAIGE_shared/LibriSpeech/test-clean --output_dir test_folders/1116_AE_1_5_new_1 --use_midway --midway_t 400 --lam 0.1

# python main.py --synthesis --load_model saved_models/1124_sc_6kb/model_best.amlt  --seq_len_p_sec 3.2 --inp_channels 64 --diff_dims 256 --continuous_type VAE_2458 --discrete_type Encodec --cond_channels 128 --cond_bandwidth 9 --ratios 8 5 4 2 --use_shortcut --input_dir /N/project/SAIGE_shared/LibriSpeech/test-clean --output_dir test_folders/1124_sc_6_9 --syn_num_timesteps 2 

conda activate versa
cd /N/slate/hy17/Projects/EXTERNAL/versa

python versa/bin/scorer.py \
    --score_config egs/speech_haici.yaml \
    --gt /N/slate/hy17/Projects/LaDiffCodec/test_folders/orig \
    --pred /N/slate/hy17/Projects/LaDiffCodec/test_folders/1116_AE_1_5_new_1 \
    --output_file 1116_AE_1_5_new_1 \
    --io dir