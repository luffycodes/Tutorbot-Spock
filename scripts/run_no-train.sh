#########################################################
# Set the user parameters
FULL_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
MODEL_DIR='models'
DATA_DIR="datasets"

ALGO="kto" # choices: ["dpo", "ipo", "kto"]
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

# Preprocess the traing and evaluation data
python src/preprocess/preprocess_sft_data.py --data_dir $DATA_DIR

DPO_MODEL_PATH="${MODEL_DIR}_dpo/${MODEL_NAME}_bio-tutor_${ALGO}"

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
