#!/bin/bash

for group in 0 1
    do
        sbatch ood_training.slurm --group $group;
        echo "Submitted job for group = $group"
    done
