import csv
import ast
import json
import argparse

def validate_gpt_json(data):
    try: 
        evaluation_of_student_response = data.get("Evaluation of Student Response")
        action_based_on_evaluation = data.get("Action Based on Evaluation")
        subproblem_state = data.get("Subproblem State")
        subproblem = data.get("Subproblem")
        tutorbot = data.get("Tutorbot")
        
        if isinstance(evaluation_of_student_response, str) and len(evaluation_of_student_response) == 1 and \
            isinstance(action_based_on_evaluation, str) and len(action_based_on_evaluation) <= 2 and \
            isinstance(subproblem_state, str) and len(subproblem_state) == 1 and \
            isinstance(subproblem, str) and \
            isinstance(tutorbot, str):
            return True
        else:
            return False
    except:
        return False

def validate_spock_json(data):
    try:
        # Invalidate incomplete responses
        if "Action Based on Evaluation" not in data or \
            "Evaluation of Student Response" not in data or \
            "Subproblem State" not in data or \
            "Subproblem" not in data or \
            "Tutorbot" not in data:
            return False

        # Invalidate empty responses
        if data["Action Based on Evaluation"] == "" and \
            data["Evaluation of Student Response"] == "" and \
            data["Subproblem State"] == "" and \
            data["Subproblem"] == "" and \
            data["Tutorbot"] == "":
            return False
            
        return True
    except:
        return False

def format_dpo_dataset(response_file):
    dpo_dataset = {
        "prompt":[],
        "chosen":[],
        "rejected":[]
    }

    retained = 0
    total = 0
    
    with open(response_file, 'r') as file:
        reader = csv.reader(file) # ['prompt', 'response', 'gpt']
        next(reader)

        for row in reader:
            total += 1
            
            prompt = row[0]
            tutorbot = row[1]
            gpt = row[2]

            # Skip invalid gpt responses
            gpt = ast.literal_eval(gpt)
            if validate_gpt_json(gpt) == False:
                continue

            # Add invalid JSON string responses
            try:
                tutorbot = ast.literal_eval(tutorbot)
            except:
                dpo_dataset["prompt"].append(prompt)
                dpo_dataset["chosen"].append(str(gpt))
                dpo_dataset["rejected"].append(tutorbot)
                retained += 1
                continue

            # Add incomplete JSON string responses
            if validate_spock_json(tutorbot) == False:
                dpo_dataset["prompt"].append(prompt)
                dpo_dataset["chosen"].append(str(gpt))
                dpo_dataset["rejected"].append(str(tutorbot)) 
                retained += 1
                continue

            # Add JSON string responses that do not have the same fields
            gpt_eval = gpt["Evaluation of Student Response"].strip()
            tutorbot_eval = tutorbot["Evaluation of Student Response"].strip()

            if gpt_eval != tutorbot_eval:
                dpo_dataset["prompt"].append(prompt)
                dpo_dataset["chosen"].append(str(gpt))
                dpo_dataset["rejected"].append(str(tutorbot))
                retained += 1
                continue
            
            gpt_action = gpt["Action Based on Evaluation"].strip()
            tutorbot_action = tutorbot["Action Based on Evaluation"].strip()

            if gpt_action != tutorbot_action:
                dpo_dataset["prompt"].append(prompt)
                dpo_dataset["chosen"].append(str(gpt))
                dpo_dataset["rejected"].append(str(tutorbot))
                retained += 1
                continue

            gpt_sub_state = gpt["Subproblem State"].strip()
            tutorbot_sub_state = tutorbot["Subproblem State"].strip()

            if gpt_sub_state != tutorbot_sub_state:
                dpo_dataset["prompt"].append(prompt)
                dpo_dataset["chosen"].append(str(gpt))
                dpo_dataset["rejected"].append(str(tutorbot))
                retained += 1
                continue

    print("# of retained data points:", retained, f"({retained / total * 100:.2f}%)")
    print("# of total data points:", total)
    
    return dpo_dataset

def store(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def parse_args():
    parser = argparse.ArgumentParser(description="Process input file paths.")
    
    parser.add_argument('--response_file', type=str, required=True, help='Path to the response file')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dpo_dataset = format_dpo_dataset(args.response_file)
    store(dpo_dataset, args.data_file)