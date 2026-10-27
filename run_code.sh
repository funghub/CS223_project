#!/bin/bash
#SBATCH --job-name=cs223_run_wk7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --output=./output/slurm_project_%j.out
#SBATCH --error=./output/slurm_project_%j.err

  
# module load conda/conda
# # activate the conda environment (use source and not just $conda activate)

# source /scratch/home/lfung/miniconda3/envs/bin/activate

start=$(date +%s)
echo "start time: $start" # Print the timestamp in seconds

echo "hostname: $HOSTNAME"

mkdir -p output # this is where output and error go


# automatically detects where Conda is installed without needing a hardcoded source path
eval "$(conda shell.bash hook)"
echo "Activating Conda Environment"
conda activate cs223_class


###########################
echo "$(pwd)"

python3 ./code/cs223_scRNAdataset_read_funs.py


###########################

conda deactivate

end=$(date +%s)

echo "end time: $end" # Print the timestamp in seconds

runtime=$((end - start))

echo "run time: $runtime" # print the run time

# run UNIX command hostname on the node, print it in output
# srun hostname
hostname

# wait for 60 secs then exit job step
# srun sleep 60
sleep 60