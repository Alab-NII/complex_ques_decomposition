import torch
import transformers
from tqdm import tqdm
from all_prompts import*
import utils

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"


model_name = "Llama-3.1-8B"
model_id = "meta-llama/Meta-{}-Instruct".format(model_name)



# load data
data_name = "2wiki" # hotpot/strategyqa/2wiki
file_path = 'data/{}_test_100.json'.format(data_name)

number_of_shot = 0
setting_type = "cot"


data = utils.read_json(file_path)
# TODO: edit here
data = data[0:5]

file_out = "outputs/3_full_output/{}_{}_{}_{}.json".format(model_name.lower(), str(number_of_shot), setting_type, data_name)


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

    #
    prompt, answer = answer_ques_cot(pipeline, input, context) # answer_ques_cot/answer_subques_zeroshot
    #
    dic_ans = {}
    dic_ans["_id"] = item["_id"]
    dic_ans["question"] = item["question"]
    dic_ans["answer"] = item["answer"]
    dic_ans["context"] = item["context"]
    dic_ans["generated_ans"] = answer
    dic_ans["user_prompt"] = prompt
    data_out.append(dic_ans)


print("Finish inference ....")
utils.write_json(data_out, file_out, indent=3)
