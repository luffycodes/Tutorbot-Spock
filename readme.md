## CLASS Meet SPOCK: An Education Tutoring Chatbot based on Learning Science Principles (Accepted at EMNLP 2023)
Arxiv Paper Link: https://arxiv.org/abs/2305.13272

Follow up work: [Pedagogical Alignment of Large Language Models](https://arxiv.org/abs/2402.05000)

Please find the CLASS slides [here](https://docs.google.com/presentation/d/1_chJlZOKrsYebXJ69-lt6dsiVZ7q_AOrfnYgqNSVRIE/edit?usp=sharing).

We train an education tutoring chatbot, Spock, on Llama-13B + Vicuna-13B weights (https://github.com/lm-sys/FastChat/) weights.
To train the chatbot, we create a synthetic dataset of mock conversations between a student and a tutor based on learning science principles like scaffolding.
We employed a specialized [prompt](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/prompts/conversation_gen/v3.txt) to generate these mock conversations using OpenAI's GPT-4 APIs.

### Inference
To use the model, first install the [fastchat](https://github.com/lm-sys/FastChat/) library, and then follow the steps here:
1. Update the [conversation.py](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/fastchat/conversation.py) from our repository in the FastChat folder.
2. Update the [inference.py](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/fastchat/inference.py) from our repository in the FastChat folder.
3. Use the [apply_delta.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/apply_delta.py) on [Spock-Bio-Llama-Diff](https://huggingface.co/luffycodes/tutorbot-spock-bio-llama-diff)  to get actual Spock weights.
      - Example: ```python3 -m fastchat.model.apply_delta  --base decapoda-research/llama-13b-hf --target tutorbot_spock_vicuna_prompt_v3  --delta luffycodes/tutorbot-spock-bio-llama-diff```
      - Also, please put ```vicuna``` in the target model name since ```conversation.py``` and ```inference.py``` check if ```vicuna``` is a substring in a model name and change conversation starter and inference prompts respectively. Note we modify ```vicuna``` prompts so you would not able to able to use original ```vicuna``` models unless you revert back changes to ```conversation.py``` and ```inference.py```.
4. Build a [biology index](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/book_index_retrieval/build_index.py) with [OpenStax Biology 2e](https://openstax.org/details/books/biology-2e) textbook. Put the generated ```os_bio_2e_index.faiss``` and the [openstax_biology_2e.csv](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/book_index_retrieval/openstax_biology_2e.csv)  in same folder as inference.py i.e. ```FastChat/fastchat``` folder.

### Creating synthetic conversation and scaffolding datasets to train Spock for subjects other than Biology
#### Example of generating conversational dataset using GPT
1. Run the [mock_con_GPTx_prompt_v3.py](https://github.com/luffycodes/Tutorbot-Spock/blob/main/gptx_datagen/mock_con_GPTx_prompt_v3.py)
      - It uses [conversation prompt v3](https://github.com/luffycodes/Tutorbot-Spock/blob/main/prompts/conversation_gen/v3.txt)
2. Remember to put [openai.organization](https://github.com/luffycodes/Tutorbot-Spock/blob/main/gptx_datagen/mock_con_GPTx_prompt_v3.py#L129) and [openai.api_key](https://github.com/luffycodes/Tutorbot-Spock/blob/main/gptx_datagen/mock_con_GPTx_prompt_v3.py#L130) in the file
3. To create a scaffolding dataset, use prompts in [folder](https://github.com/luffycodes/Tutorbot-Spock/tree/main/prompts/problem_gen)

### Training
1. Run the [create_dataset_spock.py](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/fastchat/training/create_dataset_spock.py) to create the training dataset with mock conversations in FastChat Vicuna format.
2. Use the training instructions from [fastchat](https://github.com/lm-sys/FastChat/) library.

If you use this work, please cite:
CLASS Meet SPOCK: An Education Tutoring Chatbot based on Learning Science Principles
https://arxiv.org/abs/2305.13272
```
@misc{sonkar2023class,
      title={CLASS Meet SPOCK: An Education Tutoring Chatbot based on Learning Science Principles}, 
      author={Shashank Sonkar and Lucy Liu and Debshila Basu Mallick and Richard G. Baraniuk},
      year={2023},
      eprint={2305.13272},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
