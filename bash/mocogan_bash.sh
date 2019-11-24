#!/usr/bin/env bash
#SBATCH --job-name=mocogan_conv
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=/homes/abertugli/mocogan/scripts/conv/slurm.out

cd /homes/abertugli/mocogan/mocogan_venv
source bin/activate
cd ../
module load cuda/10.0

export PYTHONPATH="${PYTHONPATH}:/homes/abertugli/mocogan"

srun python -u src/train.py