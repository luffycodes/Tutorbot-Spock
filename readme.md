## CLASS Meet SPOCK: An Education Tutoring Chatbot based on Learning Science Principles
Arxiv Paper Link: https://arxiv.org/abs/2305.13272

Model: https://huggingface.co/luffycodes/tutorbot-spock-bio-llama-diff

Dataset: https://huggingface.co/datasets/luffycodes/Tutorbot-Spock-Bio-Dataset

We train an education tutoring chatbot, Spock, on Llama-13B + Vicuna-13B weights (https://github.com/lm-sys/FastChat/) weights.
To train the chatbot, we create a synthetic dataset of mock conversations between a student and a tutor based on learning science principles like scaffolding.
We employed a specialized [prompt](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/prompts/conversation_gen/v3.txt) to generate these mock conversations using OpenAI's GPT-4 APIs.

### Inference
To use the model, first install the [fastchat](https://github.com/lm-sys/FastChat/) library, and then follow the steps here:
1. Use the [apply_delta.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/apply_delta.py) on [Spock-Bio-Llama-Diff](https://huggingface.co/luffycodes/tutorbot-spock-bio-llama-diff)  to get actual Spock weights.
- Example: ```python3 -m fastchat.model.apply_delta  --base decapoda-research/llama-13b-hf --target spock_vicuna  --delta luffycodes/tutorbot-spock-bio-llama-diff```
- Please put ```vicuna``` in the target model name somewhere since ```inference.py``` and ```conversation.py``` check if ```vicuna``` in a substring in the name and change conversation and inference prompts accordingly
2. Update the [inference.py](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/fastchat/inference.py) from our repository in the FastChat folder.
3. Update the [conversation.py](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/fastchat/conversation.py) from our repository in the FastChat folder.
4. Build a [biology index](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/book_index_retrieval/build_index.py) with [OpenStax Biology 2e](https://openstax.org/details/books/biology-2e) textbook. Put the generated ```os_bio_2e_index.faiss``` and the [openstax_biology_2e.csv](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/book_index_retrieval/openstax_biology_2e.csv)  in same folder as inference.py i.e. ```FastChat/fastchat``` folder.

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
