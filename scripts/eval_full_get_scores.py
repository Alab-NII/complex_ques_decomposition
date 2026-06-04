from tqdm import tqdm
import re
from collections import defaultdict, deque
import os
import numpy as np
import json
from official_eval import *
import utils

# Define all datasets and models
data_names = ["hotpot", "2wiki", "strategyqa"]
model_names = ["Llama-3.1-8B", "Llama-3.1-70B", "o1"]

number_of_shot = 0
setting_type = "cot" # cot/shot

def remove_part(data, ids):
    new_data = []
    for item in data:
        if item["_id"] not in ids:
            new_data.append(item)
    return new_data

hotpot_remove = ['5a8ce9ed554299585d9e3764', '5abbaed855429931dba144aa', '5adfeab455429942ec259b86', '5abbdaef554299642a094b5d', '5adc760c5542994650320d11']
strategy_remove = ['1463354ede2204c8a171', 'ca296dafd5943d07b01c', '310968736449680ee61e', 'deea92d9c38f95fef4a8', 'c144bc5e23d0944b4f1c', '83395bf81226b3f487a7', '82e32a0627566be76a90', '07e2a845709bbcb30b65']

# Dictionary to store all results: {dataset: {model: {metrics}}}
all_results = {}

# Loop through all combinations
for data_name in data_names:
    all_results[data_name] = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Processing: Model={model_name}, Dataset={data_name}")
        print(f"{'='*60}")
        
        file_out_eval = "outputs/3_full_output_scores/{}_{}_{}_{}.json".format(model_name.lower(), str(number_of_shot), setting_type, data_name)
        
        # Check if file exists
        if not os.path.exists(file_out_eval):
            print(f"File not found: {file_out_eval}")
            continue
        
        data = utils.read_json(file_out_eval)
        data_out = []
        
        metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
        
        if data_name == "hotpot":
            data = remove_part(data, hotpot_remove)
            
        if data_name == "strategyqa":
            data = remove_part(data, strategy_remove)

        print("Length Data: {}".format(len(data)))

        count_correct = 0
        for item in tqdm(data, desc=f"{model_name}-{data_name}"):
            item["pred_ans"] = str(item["pred_ans"])
            item["gold_ans"] = str(item["gold_ans"])
            
            em, f1, prec, recall = update_answer(metrics, item["pred_ans"], item["gold_ans"])
            if item["LLM-judge"] == "CORRECT":
                count_correct += 1
            
            data_out.append(item)

        # Calculate scores
        llm_judge_score = round((count_correct/len(data)*100), 1) if len(data) > 0 else 0.0
        print("This is LLM-judge evaluation")
        print(llm_judge_score)

        print("These are EM and F1 scores ....")
        N = len(data)
        if N > 0:
            for k in metrics.keys():
                metrics[k] = round(metrics[k]/N*100, 1)
        
        print(json.dumps(metrics, indent=4))
        
        # Store results
        all_results[data_name][model_name] = {
            'llm_judge': llm_judge_score,
            'em': metrics['em'],
            'f1': metrics['f1'],
            'prec': metrics['prec'],
            'recall': metrics['recall']
        }

# Print results as a formatted table
print("\n" + "="*100)
print("RESULTS TABLE - FULL END-TO-END EVALUATION")
print("="*100)

# Header
header = f"{'Dataset':<12}"
for model in model_names:
    header += f"| {model+' LLM':<13} {model+' EM':<13} {model+' F1':<13}"
print(header)
print("-" * 100)

# Rows for each dataset
for data_name in data_names:
    row = f"{data_name:<12}"
    for model_name in model_names:
        if model_name in all_results[data_name]:
            r = all_results[data_name][model_name]
            row += f"| {r['llm_judge']:<13.1f} {r['em']:<13.1f} {r['f1']:<13.1f}"
        else:
            row += f"| {'N/A':<13} {'N/A':<13} {'N/A':<13}"
    print(row)

print("="*100)