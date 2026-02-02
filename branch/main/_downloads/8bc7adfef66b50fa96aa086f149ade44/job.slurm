#!/bin/bash

#SBATCH --job-name=lightning
#SBATCH --output=./out/%x.%j.out # Note: %x == job-name
#SBATCH --error=./out/%x.%j.err # %j == job_id
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:10:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=tra26_castiel2
#SBATCH --qos boost_qos_dbg

mkdir -p ./out

module load profile/deeplrn cineca-ai/4.3.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python iris_with_datamodule.py
