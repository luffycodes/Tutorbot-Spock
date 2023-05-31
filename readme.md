# CLASS Meet SPOCK: An Education Tutoring Chatbot based on Learning Science Principles
Paper Link: https://arxiv.org/abs/2305.13272

Model: https://huggingface.co/luffycodes/tutorbot-spock-bio-llama-diff

Dataset: https://huggingface.co/datasets/luffycodes/Tutorbot-Spock-Bio-Dataset

We train an education tutoring chatbot, Spock, on Llama-13B + Vicuna weights (https://github.com/lm-sys/FastChat/) weights.
To train the chatbot, we design a specialized [prompt](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/prompts/conversation_gen/v3.txt) to create mock conversations.

To use the model, use the [fastchat](https://github.com/lm-sys/FastChat/) library by following the steps here:
1. Use the [apply_delta.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/apply_delta.py) on [Spock-Bio-Llama-Diff](https://huggingface.co/luffycodes/tutorbot-spock-bio-llama-diff)
2. Update the [inference.py](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/fastchat/inference.py) from this repo.
3. Update the [conversation.py](https://github.com/luffycodes/Tutorbot-Spock-Bio/blob/main/fastchat/conversation.py) from this repo.


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
