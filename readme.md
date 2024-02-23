## CLASS Meet SPOCK: An Education Tutoring Chatbot based on Learning Science Principles (Accepted at EMNLP 2023)
To read about more details, please refer to the arxiv paper: https://arxiv.org/abs/2305.13272 <br>
For more related informationn, find the CLASS slides [here](https://docs.google.com/presentation/d/1_chJlZOKrsYebXJ69-lt6dsiVZ7q_AOrfnYgqNSVRIE/edit?usp=sharing). <br>
To read about the follow-up work, refer to [Pedagogical Alignment of Large Language Models](https://arxiv.org/abs/2402.05000) <br>

### Overview
We aim to develop more effective educational tutoring chatbots (Spock) that help students deepen their understanding of academic subjects.
To train the chatbot, we create a synthetic dataset of mock conversations between a student and a tutor based on learning science principles like scaffolding.
We employ a specialized [prompt](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/prompts/conversation_gen/v3.txt) to generate these mock conversations using OpenAI's GPT-4 APIs.
Leveraging these conversational datasets, we supervise fine-tune the chatbots from [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [Zephyr](HuggingFaceH4/zephyr-7b-beta), and [Vicuna](lmsys/vicuna-7b-v1.5) using [FastChat](https://github.com/lm-sys/FastChat/) and then rlhf fine-tune the chatbots using [TRL](https://github.com/huggingface/trl).

### Inference
To use the models, first install the [fastchat](https://github.com/lm-sys/FastChat/) library into the root dir, and then follow the steps here:
1. Update the [conversation.py](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/fastchat/conversation.py) from our repository in the FastChat folder.
2. Update the [inference.py](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/fastchat/inference.py) from our repository in the FastChat folder.
3. Use the [apply_delta.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/apply_delta.py) on [Spock-Bio-Llama-Diff](https://huggingface.co/luffycodes/tutorbot-spock-bio-llama-diff)  to get actual model weights.
      - Example: ```python3 -m fastchat.model.apply_delta  --base decapoda-research/llama-13b-hf --target tutorbot_spock_vicuna_prompt_v3  --delta luffycodes/tutorbot-spock-bio-llama-diff```
      - Also, please put ```vicuna``` in the target model name since ```conversation.py``` and ```inference.py``` check if ```vicuna``` is a substring in a model name and change conversation starter and inference prompts respectively. Note we modify ```vicuna``` prompts so you would not able to able to use original ```vicuna``` models unless you revert back changes to ```conversation.py``` and ```inference.py```.
4. Build a [biology index](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/book_index_retrieval/build_index.py) with [OpenStax Biology 2e](https://openstax.org/details/books/biology-2e) textbook. Put the generated ```os_bio_2e_index.faiss``` and the [openstax_biology_2e.csv](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/book_index_retrieval/openstax_biology_2e.csv)  in same folder as inference.py i.e. ```FastChat/fastchat``` folder.

For easier access of the models, download them directly from Hugging Face. <br>
SFT Models:
- [mistral-7b-instruct-v0.2-class-bio-tutor-sft](https://huggingface.co/kangqi-ni/mistral-7b-instruct-v0.2-class-bio-tutor-sft)
- [zephyr-7b-beta-class-bio-tutor-sft](https://huggingface.co/kangqi-ni/zephyr-7b-beta-class-bio-tutor-sft)
- [vicuna-7b-v1.5-class-bio-tutor-sft](https://huggingface.co/kangqi-ni/vicuna-7b-v1.5-class-bio-tutor-sft)

SFT + DPO Models:
- [mistral-7b-instruct-v0.2-class-bio-tutor-dpo](https://huggingface.co/kangqi-ni/mistral-7b-instruct-v0.2-class-bio-tutor-dpo)
- [zephyr-7b-beta-class-bio-tutor-dpo](https://huggingface.co/kangqi-ni/zephyr-7b-beta-class-bio-tutor-dpo)
- [vicuna-7b-v1.5-class-bio-tutor-dpo](https://huggingface.co/kangqi-ni/vicuna-7b-v1.5-class-bio-tutor-dpo)

### Creating synthetic conversation and scaffolding datasets to train Spock for subjects other than Biology
#### Example of generating conversational dataset using GPT
1. Run the [mock_con_GPTx_prompt_v3.py](https://github.com/luffycodes/Tutorbot-Spock/blob/main/gptx_datagen/mock_con_GPTx_prompt_v3.py)
      - It uses [conversation prompt v3](https://github.com/luffycodes/Tutorbot-Spock/blob/main/prompts/conversation_gen/v3.txt)
2. Remember to put [openai.organization](https://github.com/luffycodes/Tutorbot-Spock/blob/main/gptx_datagen/mock_con_GPTx_prompt_v3.py#L129) and [openai.api_key](https://github.com/luffycodes/Tutorbot-Spock/blob/main/gptx_datagen/mock_con_GPTx_prompt_v3.py#L130) in the file
3. To create a scaffolding dataset, use prompts in [folder](https://github.com/luffycodes/Tutorbot-Spock/tree/main/prompts/problem_gen)

### Training and Evaluation
Please refer to a more detailed readme.md inside the rlhf folder for training and evaluating the models.
<!-- ### SFT (CLASS Meet SPOCK)
1. Run the [create_dataset_spock.py](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/fastchat/training/create_dataset_spock.py) to create the training dataset with mock conversations in FastChat Vicuna format.
2. Use the training instructions from [fastchat](https://github.com/lm-sys/FastChat/) library.

### RLHF (Pedagogical Alignment)
The RLHF subfolder includes the follow-up work in [Pedagogical Alignment of Large Language Models](https://arxiv.org/abs/2402.05000).
To run the entire pipeline of data preprocessing, sft, rlhf, and evaluating, download FastChat into the root folder and simply run ```./run.sh``` within the rlhf folder. Replace the MODEL_NAME or ALGO variables within the script to try out other types of models or rlhf algorithms.  -->

If you use this work, please cite: <br>
- [CLASS Meet SPOCK: An Education Tutoring Chatbot based on Learning Science Principles](https://arxiv.org/abs/2305.13272) <br>
- [Pedagogical Alignment of Large Language Models](https://arxiv.org/abs/2402.05000) <br>

```
@misc{sonkar2023class,
      title={CLASS Meet SPOCK: An Education Tutoring Chatbot based on Learning Science Principles}, 
      author={Shashank Sonkar and Lucy Liu and Debshila Basu Mallick and Richard G. Baraniuk},
      year={2023},
      eprint={2305.13272},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{sonkar2024pedagogical,
      title={Pedagogical Alignment of Large Language Models}, 
      author={Shashank Sonkar and Kangqi Ni and Sapana Chaudhary and Richard G. Baraniuk},
      year={2024},
      eprint={2402.05000},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
