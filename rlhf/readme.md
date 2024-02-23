## Pedagogical Alignment of Large Language Models
For more details, please refer to the arxiv paper: https://arxiv.org/abs/2402.05000

### Overview
This folder contains the code to supervise fine-tune and rlhf fine-tune open-sourced LLMs including [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [Zephyr](HuggingFaceH4/zephyr-7b-beta), and [Vicuna](lmsys/vicuna-7b-v1.5).

### Datasets
The training and evaluation requires bio-dataset-1.json, bio-dataset-2.json, and bio-dataset-3.json from the datasets folder.
Each contains mock conversations between a student and a tutor based on biology concepts generated from OpenAI's GPT-4.

### Pipeline
The entire pipeline is a multi-stage process: data preprocessing, sft, reward data generating, rlhf, and evaluation. Executive the script ```./run.sh``` to run this pipeline end-to-end. Replace the ```MODEL_NAME``` or ```ALGO``` variables within the script to try out other types of models or rlhf algorithms. The following sections explain each stage of the pipline separately.

#### SFT Data Preprocessing
preprocess/preprocess_sft_data.py preprocesses bio-dataset-1.json, bio-dataset-2.json, and bio-dataset-3.json into SFT, DPO, and Test splits.
Run ```python preprocess/preprocess_sft_data.py``` without additional inputs.

#### SFT
To supervise fine-tune the chatbots, install the [FastChat](https://github.com/lm-sys/FastChat/) library into the parent dir, and run the following commands. Adjust the parameters as needed.
```
FULL_MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
DATA_DIR="../datasets"
MODEL_NAME=${FULL_MODEL_PATH##*/}

SFT_DATASET_PATH="${DATA_DIR}/bio-sft.json"
SFT_MODEL_PATH="spock_bio/spock_bio_${MODEL_NAME}"

mkdir -p spock_bio
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
```

#### Reward Data Generating
Use the SFT model to generate responses on the DPO split, and then preprocess into the DPO format.
```
FULL_MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
DATA_DIR="../datasets"
MODEL_NAME=${FULL_MODEL_PATH##*/}

SFT_MODEL_PATH="spock_bio/spock_bio_${MODEL_NAME}"
DPO_DATASET_PATH="${DATA_DIR}/bio-dpo_formatted.json"
DPO_DATASET="${DATA_DIR}/${MODEL_NAME}_bio_dpo.json"

CUDA_VISIBLE_DEVICES=0 python evaluate/generate_responses.py --model_path $SFT_MODEL_PATH --output_dir ${SFT_MODEL_PATH}/final_checkpoint-dpo --test_dataset_path $DPO_DATASET_PATH

python preprocess/preprocess_dpo_data.py --response_file ${SFT_MODEL_PATH}/final_checkpoint-dpo/responses.csv --data_file $DPO_DATASET
```

#### RLHF
Install the [TRL](https://github.com/huggingface/trl) library, to perform RLHF on the SFT model.
```
FULL_MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
DATA_DIR="../datasets"
MODEL_NAME=${FULL_MODEL_PATH##*/}
ALGO="dpo"
BETA="0.1"

SFT_MODEL_PATH="spock_bio/spock_bio_${MODEL_NAME}"
DPO_MODEL_PATH="spock_bio_dpo/spock_bio_dpo_${MODEL_NAME}"
DPO_DATASET="${DATA_DIR}/${MODEL_NAME}_bio_dpo.json"

mkdir -p spock_bio_dpo

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file=ds_config/deepspeed_zero3.yaml --num_processes=4 train/train.py \
    --train_data $DPO_DATASET \
    --model_path $SFT_MODEL_PATH \
    --output_dir $DPO_MODEL_PATH \
    --beta $BETA \
    --loss $ALGO \
    --gradient_checkpointing \
    --bf16
```

#### Evaluation
Evaluate the models' performance on the test data split.
```
FULL_MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME=${FULL_MODEL_PATH##*/}
SFT_MODEL_PATH="spock_bio/spock_bio_${MODEL_NAME}"
DPO_MODEL_PATH="spock_bio_dpo/spock_bio_dpo_${MODEL_NAME}"
TEST_DATASET_PATH="${DATA_DIR}/bio-test_formatted.json"

# Generate responses from the SFT model
CUDA_VISIBLE_DEVICES=0 python evaluate/generate_responses.py --model_path $SFT_MODEL_PATH --output_dir ${SFT_MODEL_PATH}/final_checkpoint-eval --test_dataset_path $TEST_DATASET_PATH

# Generate responses from the RLHF model
CUDA_VISIBLE_DEVICES=0 python evaluate/generate_responses.py --model_path $DPO_MODEL_PATH --output_dir ${DPO_MODEL_PATH}/final_checkpoint-eval --test_dataset_path $TEST_DATASET_PATH

# Evaluate the SFT model
echo "Metrics of the SFT Model:"
python evaluate/evaluate_responses.py --response_file ${SFT_MODEL_PATH}/final_checkpoint-eval/responses.csv

# Evaluate the RLHF model
echo "Metrics of the RLHF Model:"
python evaluate/evaluate_responses.py --response_file ${DPO_MODEL_PATH}/final_checkpoint-eval/responses.csv
```
