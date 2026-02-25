RDZVID=$RANDOM
CUDA_LAUNCH_BLOCKING=1

torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    --rdzv_id $RDZVID \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:64425" \
    rlgssl.py