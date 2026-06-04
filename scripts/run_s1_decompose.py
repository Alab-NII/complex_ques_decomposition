import torch
import transformers
from tqdm import tqdm
from all_prompts import*

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4, 5"


# For 3.1
model_name = "Llama-3.1-70B"
model_id = "meta-llama/Meta-{}-Instruct".format(model_name)

# from my files
from utils import read_json, write_json, remove_duplicates, extract_sub_questions

# load data
data_name = "2wiki" # hotpot/strategyqa
number_of_shot = 2
setting_type = "shot"

# TODO: edit the path to your files
file_path = 'data/{}_test_100.json'.format(data_name)


# TODO: edit the paths here
file_demonstration = "data/demonstrations/{}_{}_demonstrations.json".format(data_name, str(number_of_shot))
file_out = "outputs/1_decomposition_output/{}_{}_{}_{}.json".format(model_name.lower(), str(number_of_shot), setting_type, data_name)


data = read_json(file_path)
data_demonstration = read_json(file_demonstration)
examples = data_demonstration


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

data_out = []


for item in tqdm(data):
    #
    input = item["question"]
    context = item["context"]
    prompt, answer = generate_subques_fewshot(pipeline, input, context, examples)
    # parse the answer to get sub-questions
    subquestions = remove_duplicates(extract_sub_questions(answer))
    #
    dic_ans = {}
    dic_ans["_id"] = item["_id"]
    dic_ans["question"] = input
    dic_ans["generated_ans"] = answer
    dic_ans["user_prompt"] = prompt
    dic_ans["answer"] = item["answer"]
    dic_ans["decomposition"] = item["decomposition"]
    dic_ans["context"] = item["context"]
    dic_ans["subquestions"] = subquestions
    data_out.append(dic_ans)


print("Finish inference ....")
write_json(data_out, file_out, indent=3)
