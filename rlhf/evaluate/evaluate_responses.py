import csv
import ast
import json
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def parse_args():
    parser = argparse.ArgumentParser(description="Process the input parameters.")
    parser.add_argument('--response_file', type=str, required=True, help="Path to the response file.")
    args = parser.parse_args()

    return args

def validate_gpt_json(data):
    try:             
        evaluation_of_student_response = data.get("Evaluation of Student Response")
        action_based_on_evaluation = data.get("Action Based on Evaluation")
        subproblem_state = data.get("Subproblem State")
        subproblem = data.get("Subproblem")
        tutorbot = data.get("Tutorbot")
        
        if (
            isinstance(evaluation_of_student_response, str) and len(evaluation_of_student_response) == 1 and 
            isinstance(action_based_on_evaluation, str) and len(action_based_on_evaluation) <= 2 and 
            isinstance(subproblem_state, str) and len(subproblem_state) == 1 and 
            isinstance(subproblem, str) and 
            isinstance(tutorbot, str)
        ):
            return True
        else:
            return False
    except json.JSONDecodeError:
        return False

def validate_spock_json(data):
    try:
        if "Evaluation of Student Response" not in data or \
            "Action Based on Evaluation" not in data or \
            "Subproblem State" not in data or \
            "Subproblem" not in data or \
            "Tutorbot" not in data:
            return False
        return True
    except:
        return False

def compute_metrics(data, title):    
    print(title)
    data['pred'] = [r if r != None else '' for r in data['pred']]
    accuracy = accuracy_score(data["actual"], data["pred"])
    print(f"Accuracy: {accuracy:.4f}")

    labels= np.unique(data['actual'] + data['pred'])
    precision, recall, f1_scores, _ = precision_recall_fscore_support(data["actual"], data["pred"], labels=labels, zero_division=0)
    macro_f1 = np.mean(f1_scores[1:])
    print(f"Macro F1 score: {macro_f1:.4f}\n")

if __name__ == "__main__":
    args = parse_args()
    
    evaluations = {"pred":[], "actual":[]}
    actions = {"pred":[], "actual":[]}
    subproblem_states = {"pred":[], "actual":[]}
    
    with open(args.response_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
    
            # Empty the tutorbot response if it is corrupted
            try:
                tutorbot = ast.literal_eval(row[1])
            except: 
                tutorbot = {
                    "Evaluation of Student Response":"",
                    "Action Based on Evaluation":"",
                    "Subproblem State":"",
                    "Tutorbot":""
                }
            
            gpt = ast.literal_eval(row[2])
    
            # Skip the response if the gpt response is corrupted
            if validate_gpt_json(gpt) == False:
                continue
    
            # Add evaluations
            try:
                evaluations["pred"].append(tutorbot['Evaluation of Student Response'])
            except:
                evaluations["pred"].append('')
                
            evaluations["actual"].append(gpt['Evaluation of Student Response'])
    
            try:
                actions["pred"].append(tutorbot['Action Based on Evaluation'])
            except:
                actions["pred"].append('')
    
            actions["actual"].append(gpt['Action Based on Evaluation'])
    
            try:
                subproblem_states["pred"].append(tutorbot['Subproblem State'])
            except:
                subproblem_states["pred"].append('')
                
            subproblem_states["actual"].append(gpt['Subproblem State'])
    
    compute_metrics(evaluations, 'Evaluation of Student Response')
    compute_metrics(actions, 'Action Based on Evaluation')
    compute_metrics(subproblem_states, 'Subproblem State')