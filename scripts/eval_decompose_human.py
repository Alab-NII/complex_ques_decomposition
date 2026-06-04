import utils

# Define all datasets
data_names = ["hotpot", "2wiki", "strategyqa"]
names = ["an01", "an02"] 

# Dictionary to store all results: {dataset: {model: {metric: score}}}
all_results = {}

# Loop through all datasets
for data_name in data_names:
    print(f"\n{'='*60}")
    print(f"Processing Dataset: {data_name}")
    print(f"{'='*60}")
    
    data = []
    for name in names:
        # TODO: edit the path to your files
        file_1 = "human_evaluation/human_judgement/{}_file_{}.json".format(data_name, name)
        try:
            data += utils.read_json(file_1)
        except:
            print(f"File not found: {file_1}")
            continue

    perfect_gpt = 0.0
    reason_chain_gpt = 0.0
    sub_num_gpt = 0.0

    perfect_70b = 0.0
    reason_chain_70b = 0.0
    sub_num_70b = 0.0

    perfect_8b = 0.0
    reason_chain_8b = 0.0
    sub_num_8b = 0.0

    list_remove = []

    data_use = data

    print("Data Length: {}".format(len(data_use)))

    count_use = 0
    for item in data_use:
        if item["is gold decomposition good or not"] != 0:
            # print(item["_id"])
            count_use += 1
            # Ensure annotations are present
            assert item["eval_gpt"]["perfect decomposition"] != -1, "Miss annotation"
            assert item["eval_70b"]["perfect decomposition"] != -1, "Miss annotation"
            assert item["eval_8b"]["perfect decomposition"] != -1, "Miss annotation"

            # Process GPT
            perfect_gpt += item["eval_gpt"]["perfect decomposition"]
            if item["eval_gpt"]["perfect decomposition"] == 1:
                reason_chain_gpt += 1
                sub_num_gpt += 1
            else:
                reason_chain_gpt += item["eval_gpt"]["reasoning chain"]
                array_re = item["eval_gpt"]["number-correct-sub-questions"]
                sub_num_gpt += sum(array_re) / len(array_re)

            # Process 70B
            perfect_70b += item["eval_70b"]["perfect decomposition"]
            if item["eval_70b"]["perfect decomposition"] == 1:
                reason_chain_70b += 1
                sub_num_70b += 1
            else:
                reason_chain_70b += item["eval_70b"]["reasoning chain"]
                array_re = item["eval_70b"]["number-correct-sub-questions"]
                sub_num_70b += sum(array_re) / len(array_re)

            # Process 8B
            perfect_8b += item["eval_8b"]["perfect decomposition"]
            if item["eval_8b"]["perfect decomposition"] == 1:
                reason_chain_8b += 1
                sub_num_8b += 1
            else:
                reason_chain_8b += item["eval_8b"]["reasoning chain"]
                array_re = item["eval_8b"]["number-correct-sub-questions"]
                sub_num_8b += sum(array_re) / len(array_re)

        else:
            list_remove.append(item["_id"])

    print(list_remove)

    num_data = count_use
    print("Number of data: {}".format(num_data))
    
    # Calculate and store results
    if num_data > 0:
        eval_gpt = {"perfect": round(perfect_gpt/num_data*100, 1), "reason_chain": round(reason_chain_gpt/num_data*100, 1), "sub_num": round(sub_num_gpt/num_data*100, 1)}
        eval_70b = {"perfect": round(perfect_70b/num_data*100, 1), "reason_chain": round(reason_chain_70b/num_data*100, 1), "sub_num": round(sub_num_70b/num_data*100, 1)}
        eval_8b = {"perfect": round(perfect_8b/num_data*100, 1), "reason_chain": round(reason_chain_8b/num_data*100, 1), "sub_num": round(sub_num_8b/num_data*100, 1)}
        
        all_results[data_name] = {
            "gpt": eval_gpt,
            "70b": eval_70b,
            "8b": eval_8b
        }
        
        print("GPT Scores:", eval_gpt)
        print("70B Scores:", eval_70b)
        print("8B Scores:", eval_8b)
    else:
        print(f"No valid data for {data_name}")

# Print results as a formatted table
print("\n" + "="*130)
print("RESULTS TABLE - HUMAN EVALUATION SCORES")
print("="*130)

# Header
header = f"{'Dataset':<12}"
header += f"| {'GPT Perfect':<12} {'GPT Chain':<12} {'GPT SubNum':<12}"
header += f"| {'70B Perfect':<12} {'70B Chain':<12} {'70B SubNum':<12}"
header += f"| {'8B Perfect':<12} {'8B Chain':<12} {'8B SubNum':<12}"
print(header)
print("-" * 130)

# Rows for each dataset
for data_name in data_names:
    if data_name in all_results:
        row = f"{data_name:<12}"
        # GPT scores
        row += f"| {all_results[data_name]['gpt']['perfect']:<12.1f} {all_results[data_name]['gpt']['reason_chain']:<12.1f} {all_results[data_name]['gpt']['sub_num']:<12.1f}"
        # 70B scores
        row += f"| {all_results[data_name]['70b']['perfect']:<12.1f} {all_results[data_name]['70b']['reason_chain']:<12.1f} {all_results[data_name]['70b']['sub_num']:<12.1f}"
        # 8B scores
        row += f"| {all_results[data_name]['8b']['perfect']:<12.1f} {all_results[data_name]['8b']['reason_chain']:<12.1f} {all_results[data_name]['8b']['sub_num']:<12.1f}"
        print(row)
    else:
        row = f"{data_name:<12}| {'N/A':<12} {'N/A':<12} {'N/A':<12}| {'N/A':<12} {'N/A':<12} {'N/A':<12}| {'N/A':<12} {'N/A':<12} {'N/A':<12}"
        print(row)

print("="*130)

