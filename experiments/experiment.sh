#!/bin/bash

# Generic options:

#SBATCH --account=ngm-sgd_project # Run job under project <project>
#SBATCH --time=2-00:00:00       # Run for a max of 2 days

# Node resources:
# (choose between 1-4 gpus per node)

#SBATCH --partition=gpu    # Choose either "gpu", "test" or "infer" partition type
#SBATCH --nodes=1           # Resources from a single node
#SBATCH --gres=gpu:1        # One GPU per node (plus 25% of node CPU and RAM per GPU)

# Run commands:

nvidia-smi  # Display available gpu resources
nproc       # Display available CPU cores

# Place other commands here

python cifar10_lrSweep.py