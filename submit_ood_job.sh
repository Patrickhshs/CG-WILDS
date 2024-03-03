for group in False True
    do
        sbatch ood_training.slurm $group;
        echo "Submitted job for group = $group"
    done
