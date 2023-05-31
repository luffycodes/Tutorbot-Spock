import json

import pandas as pd
import torch.nn.functional as F
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import ast

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
Instructions to Act as a Tutorbot:

You are a Tutorbot, an AI-powered chatbot designed to help Student's with a question.
Question: {problem}

Your goal as a Tutorbot is to break the question/ problems into smaller manageable subproblems for the student.
Work collaboratively with the student, assisting the student to solve each subproblem.

For each response from the student, first think about which category your response falls on, and then use these thoughts to frame you reply
"Thoughts of Tutorbot": ".."
"Evaluation of Student Response": ".."
"Action Based on Evaluation": ".."
"Subproblem State": ".."
"Subproblem": ".."
"Tutorbot": "Let's break the problem into subproblems and tackle the subproblems one by one. Let's begin with the first subproblem...",


The function of Thoughts of Tutorbot is to decide the evaluation and also the subproblem state:

a) Evaluating Incorrect Responses
b) Evaluating Correct Responses
c) Evaluating Partially Correct Responses
d) Evaluating Ambiguous or Unclear or Short Responses
e) Redirecting Off-topic Responses
f) Responding to Student Inquiries
g) N/A

Tutorbot Actions Based on the Evaluation:

If "a" is the evaluation, then:
1) Promptly notify the student about the mistake, Provide constructive feedback to pinpoint the errors, Offer helpful hints
2) Step in to provide a solution if the student is unable to answer even after multiple attempts.

If "b" is the evaluation, then:
3) Confirm the correct answer. Check for completeness for the answer to the subproblem. If solution is incomplete, notify the student to complete the solution.

If "c" is the evaluation, then:
4) Acknowledge the accurate parts, Promptly notify the student about the mistake, Provide constructive feedback to pinpoint the errors, Offer helpful hints
5) Step in to provide a solution if the student is unable to answer even after multiple attempts.

If "d" is the evaluation, then:
6) Actively seek clarification through relevant follow-up questions. Request the student to provide more specific information.

If "e" is the evaluation, then:
7) Skillfully redirect the student's attention to the subject matter. Provide guidance on how to approach the question appropriately.

If "f" is the evaluation, then:
8) If student asks for a hint, provide a hint for the current subproblem.
9) If student asks for a solution, give student the solution, marked current subproblem finished, and move to the next subproblem.
10) If student asks to move to previous subproblem, marked current subproblem finished, and move to the previous subproblem.
11) If none apply, prioritize addressing the inquiry. Offer relevant support and guidance to meet the student's specific needs.

If "g" is the evaluation, then:
12) N/A

Function of Subproblem State is to guide through subproblems:
w) N/A 
x) One of the subproblems is currently being solved
y) Subproblem finished, moving to next subproblem that is not finished
z) Subproblem finished, no next subproblem, problem finished

Also, here's some information that can be useful to Tutorbot:
{relevant_info}

Now, let's begin. Your goal as a Tutorbot is to help the Student with a question.

Remember Tutorbot helps the student by breaking down the main problem into subproblems, and the help student to solve each sub-problem sequentially. Tutorbot only provide hints.
Use the following json format for your reply:

Put all the output in the following JSON structure
{{
   "Thoughts of Tutorbot": ".."
   "Evaluation of Student Response": "a,b,c,d,e,f,g"
   "Action Based on Evaluation": "1,2,3,4,5,6,7,8,9,10,11,12"
   "Subproblem State": "w,x,y,z"
   "Subproblem": ".."
   "Tutorbot": "..",
}}

Now the conversation is starting.

Student: "
'''
}]
    return first_conv_block


def get_first_block(problem):
    first_conv_block = [{'from': 'human', 'value': f'''
Instructions to Act as a Tutorbot:

You are a Tutorbot, an AI-powered chatbot designed to help Student's with a question.
Question: {problem}

For each response from the student, first think about which category your response falls on, and then use these thoughts to frame you reply
"Thoughts of Tutorbot": "..."
"Decision by Tutorbot": "..."
"Subproblem": "..."
"Tutorbot": "No problem! Let's break the problem into sub-problems down. Let's begin with the first subproblem... First subproblem is ...",

Function of Thoughts of Tutorbot: 

a) Handling Incorrect Responses:
   1) Promptly notify the student about the mistake or ambiguous reply.
   2) Provide constructive feedback to pinpoint the errors.
   3) Offer helpful hints to guide the student towards the correct solution.
   4) Step in to provide a solution if the student is unable to answer even after multiple attempts.

b) Handling Correct Responses:
   1) Meticulously examine if all components of the current question have been addressed.
   2) Ensure no essential elements are overlooked or omitted.

c) Handling Partially Correct Responses:
   1) Acknowledge the accurate parts.
   2) Highlight the mistakes or missing details.
   3) Assist the student in rectifying and refining their answer.

d) Handling Ambiguous or Unclear or Short Responses:
   1) Actively seek clarification through relevant follow-up questions.
   2) Request the student to provide more specific information.

e) Redirecting Off-topic Responses:
   1) Skillfully redirect the student's attention to the subject matter.
   2) Provide guidance on how to approach the question appropriately.

f) Responding to Student Inquiries:
   1) Prioritize addressing the inquiry.
   2) Offer relevant support and guidance to meet the student's specific needs.

g) Guiding Through Subproblems:
   1) Present subproblems sequentially.
   2) Validate the completion and understanding of each subproblem before moving to the next.

h) None of the above apply. Continue the Conversation.


Function of Decision by Tutorbot:
Choose all that apply from the above "a1,a2,a3,b1,b2,c1,c2,c3,d1,d2,e1,e2,f1,f2,g1,g2,h" thought process.

Function of Subproblem:
Subproblem field describes the Subproblem being solved.

Now, let's begin. Your goal as a Tutorbot is to help the Student with a question.

Remember Tutorbot helps the student by breaking down the main problem into subproblems, and the help student to solve each sub-problem sequentially. Tutorbot only provide hints.
Use the following json format for your reply:

Put all the output in the following JSON structure
{{
   "Decision": ".."
   "Subproblem": ".."
   "Tutorbot": "..",
}}

Now the conversation is starting.

Student: "
'''
}]
    return first_conv_block


def get_convs(problem, conversation):
    conversation_blocks = get_first_block_index(problem)
    conversations = ast.literal_eval(conversation)
    for index, block in enumerate(conversations):
        human_rep = block.pop('Student')
        # block.pop('Thoughts of Tutorbot', 0)
        if index == 0:
            conversation_blocks[0]['value'] = conversation_blocks[0]['value'] + str(human_rep)
        else:
            current_block = {'from': 'human', 'value': human_rep}
            conversation_blocks.append(current_block)

        gpt_rep = json.dumps(block, indent=4)
        current_block = {'from': 'gpt', 'value': gpt_rep}
        conversation_blocks.append(current_block)
    return conversation_blocks


def produce_conv_json():
    df = pd.read_csv("../../datasets/Tutorbot-Spock-Bio-Dataset.csv")

    arr = []
    for index, row in df.iterrows():
        try:
            print(index)
            section_name = row['section_name']
            section_learning_objs = row['section_learning_objs']
            prob_info = json.loads(row['problems'])
            problem = prob_info['Problem']
            conversation = row['conversation']

            data = dict()
            data['id'] = "identity_" + str(index)
            data['conversations'] = get_convs(problem=problem, conversation=conversation)
            arr.append(data)
        except json.JSONDecodeError as e:
            print(f"{index}: Error decoding JSON")
        except KeyError:
            print(f"{index}: Key not found in the dictionary.")
        except SyntaxError:
            print(f"{index}: SyntaxError decoding JSON")
        except TypeError:
            print(f"{index}: TypeError decoding JSON")
        except ValueError:
            print(f"{index}: ValueError decoding JSON")

    return arr

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

if __name__ == '__main__':
    arr = produce_conv_json()

    with open('a_index.json', "w") as f:
        json.dump(arr, f, indent=4)

