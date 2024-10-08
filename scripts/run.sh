#########################################################
# Set the user parameters
FULL_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
MODEL_DIR='models'
DATA_DIR="datasets"

SFT_OPTION="transformers" # choices: ["transformers", "fastchat"]

ALGO="dpo" # choices: ["dpo", "ipo", "kto"]
BETA=0.1 # choices: [0.0 - 1.0]
#########################################################

# Extract the model name
MODEL_NAME=${FULL_MODEL_PATH##*/}

# Set the data paths
SFT_DATASET_PATH="${DATA_DIR}/bio-sft.json"
TEST_DATASET_PATH="${DATA_DIR}/bio-test_formatted.json"
DPO_DATASET_PATH="${DATA_DIR}/bio-dpo_formatted.json"

# Set the SFT model path
SFT_MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}_bio-tutor_sft"

# Set the DPO preference data path
DPO_PREF_DATASET_PATH="${DATA_DIR}/${MODEL_NAME}_bio_dpo.json"

# Create dirs
mkdir -p $MODEL_DIR
mkdir -p "${MODEL_DIR}_dpo"

# Preprocess sft data
python src/preprocess/preprocess_sft_data.py --data_dir $DATA_DIR

# Run SFT
if [ "$SFT_OPTION" == "transformers" ]; then
    # Run SFT with Transformers
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=20001 src/train/train_sft.py \
        --model_path $FULL_MODEL_PATH \
        --train_dataset_path $SFT_DATASET_PATH \
        --eval_dataset_path ${DATA_DIR}/bio-test.json \
        --output_dir $SFT_MODEL_PATH \
        --cache_dir cache \
        --bf16 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --eval_accumulation_steps 20 \
        --save_strategy "epoch" \
        --seed 42 \
        --learning_rate 2e-5 \
        --weight_decay 0.05 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --max_seq_length 4096 \
        --gradient_checkpointing
elif [ "$SFT_OPTION" == "fastchat" ]; then
    # Run SFT with FastChat
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=20001 FastChat/fastchat/train/train.py \
        --model_name_or_path $FULL_MODEL_PATH \
        --data_path $SFT_DATASET_PATH \
        --eval_data_path ${DATA_DIR}/bio-test.json \
        --output_dir $SFT_MODEL_DIR \
        --cache_dir cache \
        --bf16 True \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 2 \
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
        --gradient_checkpointing True
else
    echo "Invalid SFT_OPTION value. Please set SFT_OPTION to 'transformers' or 'fastchat'."
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/evaluate/generate_responses.py --model_path $SFT_MODEL_PATH --output_dir ${SFT_MODEL_PATH}/final_checkpoint-dpo --test_dataset_path $DPO_DATASET_PATH --batch_size 256

# Preprocess alignment data
python src/preprocess/preprocess_dpo_data.py --response_file ${SFT_MODEL_PATH}/final_checkpoint-dpo/responses.csv --data_file $DPO_PREF_DATASET_PATH

DPO_MODEL_PATH="${MODEL_DIR}_dpo/${MODEL_NAME}_bio-tutor_${ALGO}"

# Run Preference Optimization
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file=ds_config/deepspeed_zero3.yaml --num_processes=4 src/train/train_dpo.py \
    --train_data $DPO_PREF_DATASET_PATH \
    --model_path $SFT_MODEL_PATH \
    --output_dir $DPO_MODEL_PATH \
    --beta $BETA \
    --loss $ALGO \
    --gradient_checkpointing \
    --bf16 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 2 \
    --num_train_epochs 3

# Generate responses from the SFT model
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/evaluate/generate_responses.py --model_path $SFT_MODEL_PATH --output_dir ${SFT_MODEL_PATH}/final_checkpoint-eval --test_dataset_path $TEST_DATASET_PATH --batch_size 256

# Generate responses from the Aligned model
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/evaluate/generate_responses.py --model_path $DPO_MODEL_PATH --output_dir ${DPO_MODEL_PATH}/final_checkpoint-eval --test_dataset_path $TEST_DATASET_PATH --batch_size 256

# Evaluate the SFT model
echo "Metrics of the SFT Model:"
python src/evaluate/evaluate_responses.py --response_file ${SFT_MODEL_PATH}/final_checkpoint-eval/responses.csv

# Evaluate the Aligned model
echo "Metrics of the Aligned Model:"
python src/evaluate/evaluate_responses.py --response_file ${DPO_MODEL_PATH}/final_checkpoint-eval/responses.csv
