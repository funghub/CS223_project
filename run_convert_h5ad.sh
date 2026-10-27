#!/bin/bash
#SBATCH --job-name=cs223_convert_h5ad_64GB_6cpus
#SBATCH --mail-user=lawrence.fung@sjsu.edu --mail-type=END,FAIL
#SBATCH --partition=nodes       # (Use your cluster's specific high-mem partition name if required)
##SBATCH --partition=himem      # (Use your cluster's specific high-mem partition name if required)
##SBATCH --nodelist=himem01     # Optional: target a specific himem node, or let Slurm pick
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --output=./output/slurm_project_%j.out
#SBATCH --error=./output/slurm_project_%j.err

# old specifications for nodes, but for this task for convert to h5ad, need more specs
# --cpus-per-task=4
# --mem-per-cpu=8000

# with himem
# --cpus-per-task=8
# --mem=128G


# module load conda/conda
# # activate the conda environment (use source and not just $conda activate)

# source /scratch/home/lfung/miniconda3/envs/bin/activate

start=$(date +%s)
echo "start time: $start" # Print the timestamp in seconds
echo "Job Name: $SLURM_JOB_NAME"
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