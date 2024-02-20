python preprocess/preprocess_dpo_data.py --response_file /data/kn22/spock_bio/spock_bio_Mistral-7B-Instruct-v0.2_r1/final_checkpoint-dpo/responses.csv --data_file /data/kn22/dpo_data/bio/dpo_train/Mistral-7B-Instruct-v0.2_r1_dpo_bio_uniform_batch_123_filtered_dpo-both.json

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file=/home/kn22/dpo/ds_config/deepspeed_zero3.yaml --num_processes=4 train.py --train_data /data/kn22/dpo_data/bio/dpo_train/zephyr-7b-beta_r1_dpo_bio_uniform_batch_123_filtered_dpo-all.json --model_path /data/kn22/spock_bio/spock_bio_zephyr-7b-beta_r1 --output_dir /data/kn22/spock_bio_dpo/spock_bio_zephyr-7b-beta_dpo_r1 --beta 0.1 --loss dpo --gradient_checkpointing --bf16

CUDA_VISIBLE_DEVICES=0 python evaluate/generate_responses.py --model_path /data/kn22/spock_bio_dpo/spock_bio_zephyr-7b-beta_dpo_r1 --output_dir /data/kn22/spock_bio_dpo/spock_bio_zephyr-7b-beta_dpo_r1/efinal_checkpoint-eval --test_dataset_path /data/kn22/dpo_data/bio/dpo_bio_uniform_batch_123_filtered_test.json

python evaluate/evaluate_responses.py --response_file /data/kn22/spock_bio/vicuna-eval/responses.csv