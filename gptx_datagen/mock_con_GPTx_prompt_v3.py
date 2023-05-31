from datasets import load_dataset
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai.error import RateLimitError, Timeout, APIError, ServiceUnavailableError, InvalidRequestError
import argparse
from datetime import datetime
from tqdm import tqdm
import json
import os
import json


def prepare_prompt(problem):
    final_query = [
        {"role": "system", "content": ""}
        ,
        {"role": "user", "content": f'''
Your goal is to create a mock conversation between Student and a Tutorbot, an AI-powered chatbot designed to help Student's with a question:
Question: {problem}

"Student": "Q. {problem}",
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

Now, let's begin. Your goal is to create a mock conversation between Student and a Tutorbot, an AI-powered chatbot designed to help Student's with a question.

Please create a mock conversation now. Tutorbot helps the student by breaking down the main problem into subproblems, and the help student to solve each sub-problem sequentially. Tutorbot only provide hints.
Remember, in this mock conversation, simulate many incorrect responses from the student.
Use the following json format:

Put all the output in the following JSON structure
[{{
   "Student": "..",
   "Thoughts of Tutorbot": ".."
   "Evaluation of Student Response": "a,b,c,d,e,f,g"
   "Action Based on Evaluation": "1,2,3,4,5,6,7,8,9,10,11,12"
   "Subproblem State": "w,x,y,z"
   "Subproblem": ".."
   "Tutorbot": "..",
}},
Repeat above N times.
]

Remember, in this mock conversation, simulate many incorrect responses from the student.
'''}
    ]
    return final_query


def run_gpt4(prompt, engine):
    try:
        response = openai.ChatCompletion.create(model=engine, messages=prompt, max_tokens=2048)
    except (RateLimitError, Timeout, APIError, ServiceUnavailableError) as exc:
        return run_gpt4(prompt, engine)
    return response


def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="gpt-4-0314", help="engines of the GPT3 model for training")
    parser.add_argument("--run_name", type=str, default="curie-context-zero", help="Name of the Run (Used in storing the model)")
    parser.add_argument("--input_choice", type=str, default='context', help="Choice of prefix used for the input construction")
    parser.add_argument("--prompt_choice", type=str, default='zero-shot', help="Choice of prompting method: zero/few-shot")
    parser.add_argument("--max_tokens", type=int, default=6, help="maximum number of token generated")
    parser.add_argument('--temp', type=float, default=0.9, help='temperature for generation (higher=more diverse)')
    parser.add_argument('--top_p', type=int, default=1, help='p value for sampling')
    parser.add_argument("--n", type=int, default=1, help="number of samples to generate")
    parser.add_argument("--best_of", type=int, default=1, help="Best of N samples to generate")

    params = parser.parse_args()
    return params


if __name__ == '__main__':
    args = add_params()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    openai.organization = ""
    openai.api_key = ""

    x = openai.Model.list()
    y = [item['root'] for item in x['data']]
    y.sort()
    for item in y:
        print(item)

    learning_obj_df = pd.read_csv('../datasets/bio-problems-learning_objs.csv.csv')

    new_df = pd.DataFrame()

    for index, row in learning_obj_df.iterrows():
        print(index)
        new_df.loc[index, 'section_name'] = row['section_name']
        new_df.loc[index, 'section_learning_objs'] = row['section_learning_objs']
        new_df.loc[index, 'section_id'] = row['section_id']
        new_df.loc[index, 'problems'] = row['problems']

        section_name = row['section_name']
        section_learning_objs = row['section_learning_objs']
        try:
            prob_info = json.loads(row['problems'])
            prompt = prepare_prompt(problem=prob_info['Problem'])
            response = run_gpt4(prompt, args.engine)
            resp = response['choices'][0]['message']['content'].strip()
            new_df.loc[index, 'conversation'] = resp
        except json.JSONDecodeError as e:
            new_df.loc[index, 'conversation'] = '[{}]'
            print(f"{index}: Error decoding JSON")

        new_df.to_csv('Tutorbot-Spock-Bio-Dataset.csv', index=False)

