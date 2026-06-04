import torch
import transformers
from tqdm import tqdm
from all_prompts import*

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# For 3.1
model_name = "Llama-3.1-8B"
model_id = "meta-llama/Meta-{}-Instruct".format(model_name)


import utils

# load data
data_name = "strategyqa" # hotpot/strategyqa
file_path = 'data/{}_test.json'.format(data_name)

number_of_shot = 0
setting_type = "shot"

data = utils.read_json(file_path)
# TODO: edit here
data = data[0:5]

file_out = "outputs/2_ans_sub_output/{}_{}_{}_{}.json".format(model_name.lower(), str(number_of_shot), setting_type, data_name)


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

data_out = []


def get_all_subques(decomposition):
    questions = []
    for item in decomposition:
        if "#1" in item["question"]:
            item["question"] = item["question"].replace("#1", decomposition[0]["answer"])
        if "#2" in item["question"]:
            item["question"] = item["question"].replace("#2", decomposition[1]["answer"])
        if "#3" in item["question"]:
            item["question"] = item["question"].replace("#3", decomposition[2]["answer"])
        questions.append(item["question"])
    return questions, decomposition


for item in tqdm(data):
    #
    context = item["context"]
    decomposition = item["decomposition"]
    questions, new_decomposition = get_all_subques(decomposition)
    for idx, ques in enumerate(questions):
        #
        prompt, answer = answer_subques_zeroshot(pipeline, ques, context)
        new_decomposition[idx]["generated_ans"] = answer
        new_decomposition[idx]["user_prompt"] = prompt
    #
    dic_ans = {}
    dic_ans["_id"] = item["_id"]
    dic_ans["decomposition"] = new_decomposition
    dic_ans["context"] = item["context"]
    data_out.append(dic_ans)


print("Finish inference ....")
utils.write_json(data_out, file_out, indent=3)
