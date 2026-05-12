BASE_DIR=$(dirname "$0")
cd "$BASE_DIR"

export CUDA_VISIBLE_DEVICES=0,1

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

PORT=${MASTER_PORT:-29500}

echo "Starting distributed training on $NUM_GPUS GPUs (Port: $PORT)..."

torchrun --standalone --nproc_per_node=$NUM_GPUS --master_port=$PORT main.py --mode train --config config.yaml 

