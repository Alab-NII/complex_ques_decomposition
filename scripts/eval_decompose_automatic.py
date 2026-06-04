from tqdm import tqdm
import re
from collections import defaultdict, deque
import os
import numpy as np
# from my files 
import utils


def build_graph(sub_questions, special_key="#"):
    graph = defaultdict(set)  # Adjacency list representation
    # Node labels (1-based indexing)
    for sub in sub_questions:
        if "#Ans" in sub:
            special_key = "#Ans"
            break
    #
    nodes = {f"{special_key}{i+1}": i for i in range(len(sub_questions))}
    #
    # Create edges based on references
    for i, question in enumerate(sub_questions):
        for ref in nodes:
            if ref in question and nodes[ref] != i:  # Avoid self-loops
                graph[i].add(nodes[ref])
                graph[nodes[ref]].add(i)
    #
    return graph


def is_connected(graph, num_nodes):
    """ Check if the graph is fully connected using BFS """
    if num_nodes == 0:
        return False
    #
    visited = set()
    queue = deque([0])  # Start from the first node
    #
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(graph[node] - visited)  # Add unvisited neighbors
    #
    return len(visited) == num_nodes  # Check if all nodes were visited


# Define all model names and dataset names
model_names = ["Llama-3.1-8B", "Llama-3.1-70B", "o1"]
data_names = ["hotpot", "2wiki", "strategyqa"]

setting_type = "shot"
number_of_shot = 2

# for new dataset format 
def obtain_gold_subques(decomposition, data_name):
    result = []
    for item in decomposition:
        result.append(item["question"])
    return result

def remove_part(data, ids):
    new_data = []
    for item in data:
        if item["_id"] not in ids:
            new_data.append(item)
    return new_data

hotpot_remove = ['5a8ce9ed554299585d9e3764', '5abbaed855429931dba144aa', '5adfeab455429942ec259b86', '5a716eeb5542994082a3e82e', '5abbdaef554299642a094b5d', '5adc760c5542994650320d11']

strategy_remove = ['f353cbd3286ea1e452aa', '1463354ede2204c8a171', 'ca296dafd5943d07b01c', '310968736449680ee61e', 'deea92d9c38f95fef4a8', 'ca09aea1e9589ef40c61', 'ecc4a9173c24ace0215a', 'c144bc5e23d0944b4f1c', '83395bf81226b3f487a7', '82e32a0627566be76a90', '07e2a845709bbcb30b65']

# Dictionary to store all results: {dataset: {model: (length_score, link_score)}}
results = {data_name: {} for data_name in data_names}

# Loop through all combinations of model names and dataset names
for model_name in model_names:
    for data_name in data_names:
        print(f"\n{'='*60}")
        print(f"Processing: Model={model_name}, Dataset={data_name}")
        print(f"{'='*60}")
        # TODO: edit the path to your prediction files
        file_pred = "outputs/1_decomposition_output/{}_{}_{}_{}.json".format(model_name.lower(), str(number_of_shot), setting_type, data_name)
        
        # Check if file exists
        if not os.path.exists(file_pred):
            print(f"File not found: {file_pred}")
            continue
        
        data_pred = utils.read_json(file_pred)
        
        if data_name == "hotpot":
            data_pred = remove_part(data_pred, hotpot_remove)
            
        if data_name == "strategyqa":
            data_pred = remove_part(data_pred, strategy_remove)

        data_use = data_pred

        # data_use = []
        # for item in data_pred:
        #     if len(item["decomposition"]) == 4:
        #         data_use.append(item)


        print("Data Length: {}".format(len(data_use)))

        scores_length = []
        scores_link = []

        for item in tqdm(data_use, desc=f"{model_name}-{data_name}"):
            # score_item = []
            pred_decompose = item["subquestions"]
            gold_decompose = obtain_gold_subques(item["decomposition"], data_name)
            # consider length in evaluation
            if len(pred_decompose) == len(gold_decompose):
                scores_length.append(1)
            else:
                scores_length.append(0)
                
            # everything is connected --- checking 
            graph = build_graph(pred_decompose)
            connected = is_connected(graph, len(pred_decompose))
            if connected == True:
                scores_link.append(1)
            else:
                scores_link.append(0)
                
        ave_length = (sum(scores_length) / len(scores_length))*100
        ave_link = (sum(scores_link) / len(scores_link))*100
        
        # Store results
        results[data_name][model_name] = (ave_length, ave_link)
        
        print(f"Average Length: {ave_length:.1f}")
        print(f"Average Link: {ave_link:.1f}")

# Print results as a formatted table
print("\n" + "="*100)
print("RESULTS TABLE")
print("="*100)

# Header
header = f"{'Dataset':<15}"
for model in model_names:
    header += f"| {model+' Len':<12} {model+' Link':<12}"
print(header)
print("-" * 100)

# Rows for each dataset
for data_name in data_names:
    row = f"{data_name:<15}"
    for model_name in model_names:
        if model_name in results[data_name]:
            length_score, link_score = results[data_name][model_name]
            row += f"| {length_score:<12.1f} {link_score:<12.1f}"
        else:
            row += f"| {'N/A':<12} {'N/A':<12}"
    print(row)

print("="*100)