#!/bin/bash
#SBATCH --job-name=wilds_job
#SBATCH --output=outputs/output_%j.txt
#SBATCH --error=errors/error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 1-05:00:00
#SBATCH --mem=48GB
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu

#GREENE GREENE_GPU_MPS=yes
singularity exec --nv --bind /scratch/$USER --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw /scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif bash -c '
source /ext3/env.sh
conda activate WILDS

python /scratch/js12556/CG-WILDS/main.py --group 1 --epoch 12 --subset_size 0.05
'