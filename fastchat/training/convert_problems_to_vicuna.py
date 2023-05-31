import json

import faiss
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                              min=1e-9)

def get_relevant_para(prompt):
    dataframe = pd.read_csv('../../book_index_retrieval/openstax_biology_2e.csv')
    dataframe = dataframe[dataframe['p_id'].str.startswith('fs-').fillna(False)]
    paragraphs = dataframe['p_content'].tolist()
    index = faiss.read_index('../../book_index_retrieval/os_bio_2e_index.faiss')

    encoded_query = tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        query_output = model(**encoded_query)
    query_embedding = mean_pooling(query_output, encoded_query['attention_mask'])
    normalized_query_embedding = F.normalize(query_embedding, p=2, dim=1)

    # Perform a search using Faiss
    k = 3  # Number of nearest neighbors to retrieve
    distances, indices = index.search(normalized_query_embedding.squeeze().unsqueeze(dim=0).numpy(), k)

    # Get the relevant paragraphs
    relevant_paragraphs = [paragraphs[i] for i in indices[0]]
    info = "\nHelpful Information for Tutorbot: "
    for para in relevant_paragraphs:
        info = info + "\n" + str(para)
    info = info + "\n End of Helpful Information for Tutorbot.\n"

    return info


def get_first_block_index(problem):
    relevant_info = get_relevant_para(problem)

    first_conv_block = [{'from': 'human', 'value': f'''        
You are a Tutorbot, an AI-powered chatbot designed to help students solve a problem.
For being a good Tutorbot, you should be able to break the problem into sequential subproblems.
Also, you should be able to predict possible incorrect student responses to each subproblem.
For the incorrect student responses, your job also involves providing necessary feedback to the student.
And ofcourse, for being a good Tutorbot, you should know the facts needed to answer the problem and most critically the solution to the problem.

Also, here's some information that can be useful to Tutorbot:
{relevant_info}

For the following Question: {problem}, now please provide the following:

1) Facts necessary to answer it,
2) Subproblems that the main problem can be broken down into, and
3) The final answer.
For each subproblem, generate a hint, one incorrect student response to the subproblem, and corresponding feedback to the student. Put all the output in the following JSON structure:
{{
    "SubProblems": [
            "Question": "..",
            "Answer": "..",
            "Hint": "..",
            "Incorrect Response": "..",
            "Feedback": ".."
    ],
    "Facts": [
        "..",
        ".."
    ],
    "Solution": ".."
}}.
 Now please provide subproblems with necessary hints, possible student incorrect responses, feedback, along with facts and solution for the problem.
 Problem: {problem}.
'''
}]
    return first_conv_block


def get_first_block(problem):
    first_conv_block = [{'from': 'human', 'value': f'''        
You are a Tutorbot, an AI-powered chatbot designed to help students solve a problem.
For being a good Tutorbot, you should be able to break the problem into sequential subproblems.
Also, you should be able to predict possible incorrect student responses to each subproblem.
For the incorrect student responses, your job also involves providing necessary feedback to the student.
And ofcourse, for being a good Tutorbot, you should know the facts needed to answer the problem and most critically the solution to the problem.
 
For the following Question: {problem}, now please provide the following:

1) Facts necessary to answer it,
2) Subproblems that the main problem can be broken down into, and
3) The final answer.
For each subproblem, generate a hint, one incorrect student response to the subproblem, and corresponding feedback to the student. Put all the output in the following JSON structure:
{{
    "SubProblems": [
            "Question": "..",
            "Answer": "..",
            "Hint": "..",
            "Incorrect Response": "..",
            "Feedback": ".."
    ],
    "Facts": [
        "..",
        ".."
    ],
    "Solution": ".."
}}.
Now please provide subproblems with necessary hints, possible student incorrect responses, feedback, along with facts and solution for the problem.
Problem: {problem}.
'''
}]
    return first_conv_block


def get_convs(problem):
    conversation_blocks = get_first_block_index(problem.pop("Problem"))
    gpt_rep = json.dumps(problem, indent=4)
    current_block = {'from': 'gpt', 'value': gpt_rep}
    conversation_blocks.append(current_block)
    return conversation_blocks


def produce_prob_json():
    df = pd.read_csv("../../datasets/bio-problems-learning_objs.csv")

    arr = []
    for index, row in df.iterrows():
        try:
            print(index)
            prob_info = json.loads(row['problems'])

            data = dict()
            data['id'] = "identity_" + str(index + 1000)
            data['conversations'] = get_convs(problem=prob_info)
            arr.append(data)
        except json.JSONDecodeError as e:
            print(f"{index}: Error decoding JSON")
        except KeyError:
            print(f"{index}: Key not found in the dictionary.")

    return  arr

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

if __name__ == '__main__':
    arr = produce_prob_json()

    with open('b.json', "w") as f:
        json.dump(arr, f, indent=4)

