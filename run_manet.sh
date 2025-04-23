#!/bin/bash
#SBATCH --job-name=ferhm
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24

# Load necessary modules (adjust as needed for your system)
#module purge
#module load cuda/11.2

# Activate your conda environment
source /data2/shared/apps/conda/etc/profile.d/conda.sh
conda activate qdhposterv2



# Go to the project directory
cd $HOME/QDHManet/MA-Net

# Run the script
python main1dfer.py
