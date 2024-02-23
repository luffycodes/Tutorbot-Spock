# Set the models and rlhf algorithms
FULL_MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
ALGO="dpo"
BETA="0.1"

# Set the dataset paths
DATA_DIR="../datasets"

MODEL_NAME=${FULL_MODEL_PATH##*/}
SFT_DATASET_PATH="${DATA_DIR}/bio-sft.json"
TEST_DATASET_PATH="${DATA_DIR}/bio-test_formatted.json"
DPO_DATASET_PATH="${DATA_DIR}/bio-dpo_formatted.json"

# Preprocess sft data
python preprocess/preprocess_sft_data.py

# Run SFT training with FastChat
mkdir -p spock_bio
SFT_MODEL_PATH="spock_bio/spock_bio_${MODEL_NAME}"

deepspeed --include localhost:0,1,2,3 --master_port=20001 ../FastChat/fastchat/train/train.py \
    --model_name_or_path $FULL_MODEL_PATH \
    --data_path $SFT_DATASET_PATH \
    --eval_data_path ${DATA_DIR}/bio-test.json \
    --output_dir $SFT_MODEL_PATH \
    --cache_dir cache \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --eval_accumulation_steps 50 \
    --save_strategy "epoch" \
    --seed 42 \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --deepspeed ../FastChat/playground/deepspeed_config_s3.json

# Generate responses from the SFT model
CUDA_VISIBLE_DEVICES=0 python evaluate/generate_responses.py --model_path $SFT_MODEL_PATH --output_dir ${SFT_MODEL_PATH}/final_checkpoint-eval --test_dataset_path $TEST_DATASET_PATH

CUDA_VISIBLE_DEVICES=0 python evaluate/generate_responses.py --model_path $SFT_MODEL_PATH --output_dir ${SFT_MODEL_PATH}/final_checkpoint-dpo --test_dataset_path $DPO_DATASET_PATH

# Preprocess dpo data
DPO_DATASET="${DATA_DIR}/${MODEL_NAME}_bio_dpo.json"
python preprocess/preprocess_dpo_data.py --response_file ${SFT_MODEL_PATH}/final_checkpoint-dpo/responses.csv --data_file $DPO_DATASET

# Run RLHF training with TRL
mkdir -p spock_bio_dpo
DPO_MODEL_PATH="spock_bio_dpo/spock_bio_dpo_${MODEL_NAME}"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file=ds_config/deepspeed_zero3.yaml --num_processes=4 train/train.py \
    --train_data $DPO_DATASET \
    --model_path $SFT_MODEL_PATH \
    --output_dir $DPO_MODEL_PATH \
    --beta $BETA \
    --loss $ALGO \
    --gradient_checkpointing \
    --bf16

# Generate responses from the RLHF models
CUDA_VISIBLE_DEVICES=0 python evaluate/generate_responses.py --model_path $DPO_MODEL_PATH --output_dir ${DPO_MODEL_PATH}/final_checkpoint-eval --test_dataset_path $TEST_DATASET_PATH

# Evaluate the SFT model
echo "Metrics of the SFT Model:"
python evaluate/evaluate_responses.py --response_file ${SFT_MODEL_PATH}/final_checkpoint-eval/responses.csv

# Evaluate the RLHF model
echo "Metrics of the RLHF Model:"
python evaluate/evaluate_responses.py --response_file ${DPO_MODEL_PATH}/final_checkpoint-eval/responses.csv