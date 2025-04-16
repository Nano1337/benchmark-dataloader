#!/bin/bash
#SBATCH --job-name=high_resource_job
#SBATCH --ntasks=1                      # Request 1 task
#SBATCH --cpus-per-task=192             # 192 CPUs for this task
#SBATCH --mem=1TB                       # Request 1 terabyte of RAM
#SBATCH --time=24:00:00                 # Request 24 hours of runtime
#SBATCH --output=results/slurm-%j.out   # Output file
#SBATCH --error=results/slurm-%j.err    # Error file

# Print some information about the job
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPU per task: $SLURM_CPUS_PER_TASK"

echo "Memory per task: $SLURM_MEM_PER_NODE"

echo "Requested time: $SLURM_TIMELIMIT"

# Load any required modules here
# module load example_module

# Set up environment variables if needed
# export EXAMPLE_VAR=value

# Change to the appropriate directory
cd $SLURM_SUBMIT_DIR

# Your commands here
# For example, to run the benchmark script:
./scripts/prepare_datasets.sh

# Print job completion time
echo "Job completed at $(date)"
