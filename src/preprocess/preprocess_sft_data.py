import json
import random
import ast
import argparse

random.seed(42)

SYSTEM_MSG = '''Instructions to Act as a Tutorbot:
You are a Tutorbot, an AI-powered expert chatbot designed to help students with a question.

Your goal as a Tutorbot is to break the question/problem into smaller manageable subproblems for the student.
Work collaboratively with the student, assisting the student to solve each subproblem.

For each response from the student, first think about which category your response falls on, and then use these thoughts to frame your reply
"Evaluation of Student Response": ".."
"Action Based on Evaluation": ".."
"Subproblem State": ".."
"Subproblem": ".."
"Tutorbot": "Let's break the problem into subproblems and tackle the subproblems one by one. Let's begin with the first subproblem...",

First, decide the Evaluation of Student Response:

a) Evaluating Incorrect Responses
b) Evaluating Correct Responses
c) Evaluating Partially Correct Responses
d) Evaluating Ambiguous or Unclear or Short Responses
e) Redirecting Off-topic Responses
f) Responding to Student Inquiries
g) N/A

Then, decide Action Based on the Evaluation:

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

Finally, decide the Subproblem State. The function of Subproblem State is to guide through subproblems:
w) N/A
x) One of the subproblems is currently being solved
y) Subproblem finished, moving to next subproblem that is not finished
z) Subproblem finished, no next subproblem, problem finished

Now, let's begin. Your goal as a Tutorbot is to help the Student with a question. Remember Tutorbot helps the student by breaking down the main problem into subproblems, and the help student to solve each sub-problem sequentially. Tutorbot only provide hints. Use the following json format for your reply: Put all the output in the following JSON structure {"Evaluation of Student Response": "a,b,c,d,e,f,g", "Action Based on Evaluation": "1,2,3,4,5,6,7,8,9,10,11,12", "Subproblem State": "w,x,y,z", "Subproblem": "..", "Tutorbot": ".."} Now the conversation is starting.

Student: '''

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for data directory")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='datasets'
    )
    return parser.parse_args()

def sample_datasets(files, num_sft_train_samples, num_dpo_train_samples, num_test_samples):
    sft_data = list()
    dpo_data = list()
    test_data = list()
    
    for file_path in files:
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Filter the data
        data = [sample for sample in data if len(sample["conversations"]) > 2]

        # Sample the data
        samples = random.sample(data, min(num_sft_train_samples + num_dpo_train_samples + num_test_samples, len(data)))

        # Partition the data
        sft_set = samples[:num_sft_train_samples]
        dpo_set = samples[num_sft_train_samples:num_sft_train_samples+num_dpo_train_samples]
        test_set = samples[num_sft_train_samples+num_dpo_train_samples:]

        # Add the data
        sft_data.extend(sft_set)
        dpo_data.extend(dpo_set)
        test_data.extend(test_set)
        
    return sft_data, dpo_data, test_data

def modify(data):
    processed_data = []
    
    for conv in data:
        processed_conv = []

        # Iterate each round
        for round in conv["conversations"]:
            # Add the instruction
            if round["from"] == "human":
                if "Instructions to Act as a Tutorbot" not in round["value"]:
                    processed_conv.append(round)
                    continue

                # Modify the system message
                idx = round["value"].find("Student: ")
                question = round["value"][idx+len("Student: "):]

                round["value"] = SYSTEM_MSG + question.replace('"\nQ.', "Q.")
                processed_conv.append(round)
            # Add the response
            else:
                try:
                    response = ast.literal_eval(round["value"])
        
                    # Remove the thought
                    response.pop('Thoughts of Tutorbot', None)

                    # Correct the tags
                    if response["Evaluation of Student Response"] == "N/A":
                        response["Evaluation of Student Response"] = "g"

                    if response["Action Based on Evaluation"] == "N/A":
                        response["Action Based on Evaluation"] = "12"

                    if response["Subproblem State"] == "N/A":
                        response["Subproblem State"] = "w"

                    # Add the string response
                    round["value"] = str(response)
                    processed_conv.append(round)
                except:
                    # Remove the previous instruction
                    processed_conv.pop()

                    # End the conversation
                    break

        # Add the conversation
        if len(processed_conv) > 0:
            processed_data.append({
                "id": conv["id"],
                "conversations": processed_conv
            })

    return processed_data

def store(data, data_file):
    with open(data_file, 'w') as json_file:
        json.dump(data, json_file)

def format_conversation(conversation):
    formatted_conversation = list()
    
    for row in conversation:
        role = row["from"]
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        else:
            exit("Unidentifiable role!")
            
        content = row["value"]
        
        formatted_conversation.append({
            "role":role,
            "content":content,
        })
        
    return formatted_conversation
    
def create_datasets(data_file_path):

    with open(data_file_path, 'r') as file:
        data = json.load(file)
    
    dataset = list()
    for row in data:
        formatted_conversation = format_conversation(row["conversations"])

        dataset.append(formatted_conversation)

    return dataset

def format_dpo(valid_set):
    global roles

    def concat(conversation):
        
        result = "System: " + conversation[0]["content"]
        result += '\n'
        
        i = 1
        for row in conversation[1:]:
            result += roles[i] + ' ' + row["content"] + '\n'
            
            i = (i + 1) % 2
        
        return result
    
    data = {"prompt":list(), "chosen":list()}
    
    for conversation in valid_set:
        for i in range(0, len(conversation), 2):
            history = conversation[:i+1]

            templated_prompt = concat(history) + "\n" + roles[1] + ''

            chosen = conversation[i+1]["content"]
            
            data["prompt"].append(templated_prompt)
            data["chosen"].append(chosen)
    return data

def convert_dpo(json_file, output_file_path):
    dataset = create_datasets(json_file)
    filtered_set = [sample for sample in dataset if len(sample) != 2]
    
    data = format_dpo(filtered_set)
    store(data, output_file_path)
    

if __name__ == "__main__":
    num_sft_train_samples = 200
    num_dpo_train_samples = 200
    num_test_samples = 150

    args = parse_args()
    
    data_dir = args.data_dir
    files = [data_dir + "/bio-dataset-1.json", data_dir + "/bio-dataset-2.json", data_dir + "/bio-dataset-3.json"]
    
    sft_train_file = data_dir + "/bio-sft.json"
    dpo_train_file = data_dir + "/bio-dpo.json"
    test_file = data_dir + "/bio-test.json"

    print("Sampling data...")
    sft_data, dpo_data, test_data = sample_datasets(files, num_sft_train_samples, num_dpo_train_samples, num_test_samples)
    
    sft_data = modify(sft_data)
    dpo_data = modify(dpo_data)
    test_data = modify(test_data)
    
    store(sft_data, sft_train_file)
    store(dpo_data, dpo_train_file)
    store(test_data, test_file)

    print("Converting the format...")
    test_file_formatted = data_dir + "/bio-test_formatted.json"
    dpo_train_file_formatted = data_dir + "/bio-dpo_formatted.json"
    
    roles = ["Student:", "Tutorbot:"]
    convert_dpo(dpo_train_file, dpo_train_file_formatted)
    convert_dpo(test_file, test_file_formatted)

    print("Done.")