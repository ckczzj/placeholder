GPUS=$1
BATCH_SIZE=${2:-256}
NUM_WORKERS=${3:-4}

torchrun \
--master_addr=127.0.0.1 --master_port=3343 \
--nnodes=1 --node_rank=0 --nproc_per_node="${GPUS}" \
main.py \
--batch_size "${BATCH_SIZE}" \
--num_workers "${NUM_WORKERS}" 
