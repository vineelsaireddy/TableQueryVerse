#!/bin/bash
#SBATCH -A research
#SBATCH -J "codellama_spider"
#SBATCH -c 40
#SBATCH --mem-per-cpu=1024
#SBATCH -G 4
#SBATCH --nodelist=gnode072
#SBATCH --time="4-00:00:00"
#SBATCH --output=Stdout.txt
#SBATCH --mail-user=kavuri.hruday@research.iiit.ac.in
#SBATCH --mail-type=ALL

source ~/.bashrc
echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"

export HF_HOME=./transformer_cache
conda activate general

python3 finetune.py

echo "Loading Model"
python3 load_model.py

echo "Time at exit: $(date)"
