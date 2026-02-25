#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Thread count:
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
# memory in MB
#SBATCH --mem=64G
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/jroth/Stable-SSL/slurm/rlgssl_pascal_voc_10_out_%a.txt
#SBATCH --error=/ourdisk/hpc/ai2es/jroth/Stable-SSL/slurm/rlgssl_pascal_voc_10_err_%a.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=rlgssl
#SBATCH --mail-user=jay.c.rothenberger@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/ourdisk/hpc/ai2es/jroth/Stable-SSL/examples/rlgssl_pascal_voc_10/
#################################################

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b")

echo Node IP: $head_node_ip

. /home/fagg/tf_setup.sh
conda activate /home/jroth/.conda/envs/tree-care-3.9

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "$head_node_ip:64425" \
rlgssl.py
