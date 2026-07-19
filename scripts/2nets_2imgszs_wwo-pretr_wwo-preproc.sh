BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"

EPOCHS="20"
LR="3e-4"
OUTPUT_DIR="./outputs_2nets-exps_e${EPOCHS}_lr${LR}"

CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
  --model-names "vgg11,efficientnet_b0" \
  --image-sizes "128,224" \
  --preprocessing-flags "true,false" \
  --pretrained-flags "true,false" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "cuda" \
  --output-dir "${OUTPUT_DIR}" \
