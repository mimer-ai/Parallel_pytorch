#!/bin/bash

#SBATCH --job-name=lightning
#SBATCH --output=./lightning_logs/%x.%j.out # Note: %x == job-name
#SBATCH --error=./lightning_logs/%x.%j.err # %j == job_id
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=tra26_castiel2
#SBATCH --reservation=s_tra_cast3

module load profile/deeplrn cineca-ai/4.3.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATA_DIR=./data
WEIGHTS_PATH=./model_weights/resnet50_imagenet.pth
LOG_DIR=./lightning_logs

# Hyperparams
BATCH_SIZE=128
NUM_WORKERS=8
LR=0.1
MAX_EPOCHS=10
STRATEGY=ddp # can be ddp, fsdp_full, fsdp_shard_grad

mkdir -p logs "$LOG_DIR"

# Choose which script to run:
SCRIPT=train_cifar10_pl_solution.py
# SCRIPT=train_cifar10_pl_student.py  # uncomment to run the student version

srun python "$SCRIPT" \
  --data_dir "$DATA_DIR" \
  --weights_path "$WEIGHTS_PATH" \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --lr $LR \
  --devices $SLURM_GPUS_PER_NODE \
  --num_nodes $SLURM_NNODES \
  --strategy "$STRATEGY" \
  --max_epochs $MAX_EPOCHS \
  --log_dir "$LOG_DIR"
