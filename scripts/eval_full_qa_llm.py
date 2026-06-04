from tqdm import tqdm
import re
from collections import defaultdict, deque
import os
import sys
import numpy as np


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4, 5"


from official_eval import *
import utils
from llm_as_a_judge_prompt import*


data_name = "2wiki" #  # hotpot/strategyqa/2wiki

number_of_shot = 0
setting_type = "cot"

qa_model_name = "Llama-3.1-8B"
# qa_model_name = "Llama-3.1-70B"
# qa_model_name = "o1"



file_out_infer = "outputs/3_full_output/{}_{}_{}_{}.json".format(qa_model_name.lower(), str(number_of_shot), setting_type, data_name)
file_out_eval = "outputs/3_full_output_scores/{}_{}_{}_{}.json".format(qa_model_name.lower(), str(number_of_shot), setting_type, data_name)

data = utils.read_json(file_out_infer)

data_out = []


# For 3.3
model_id = "meta-llama/Llama-3.3-70B-Instruct"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

count_correct = 0
metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
for item in tqdm(data):
    #
    new_item = {}
    new_item["_id"] = item["_id"]
    new_item["question"] = item["question"]
    new_item["gold_ans"] = item["answer"]
    new_item["generated_ans"] = item["generated_ans"]
    #
    new_item["pred_ans"] = utils.extract_answer(item["generated_ans"]).strip()

    #
    em, f1, prec, recall = update_answer(metrics, str(new_item["pred_ans"]), str(new_item["gold_ans"]))
    new_item["EM"] = em
    new_item["f1"] = f1
    if em == True:
        new_item["LLM-judge"] = "CORRECT"
    elif new_item["pred_ans"] == "":
        new_item["LLM-judge"] = "INCORRECT"
    elif new_item["pred_ans"].lower() == "I cannot confidently find the answer".lower():
        new_item["LLM-judge"] = "NOT_SURE"
    else:
        llm = judge_pred_answer(pipeline, item["question"], new_item["gold_ans"], new_item["pred_ans"], item["context"])
        new_item["LLM-judge"] = utils.extract_answer(llm).strip()
    data_out.append(new_item)




utils.write_json(data_out, file_out_eval, indent=3)
