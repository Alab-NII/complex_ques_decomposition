from tqdm import tqdm
import re
from collections import defaultdict, deque
import os
import sys
import re
import numpy as np

from official_eval import *
import utils
from llm_as_a_judge_prompt import*

os.environ['CUDA_VISIBLE_DEVICES'] = "6, 7"

data_name = "strategyqa" # sys.argv[1] # hotpot/strategyqa/2wiki

number_of_shot = 0
setting_type = "shot"

# TODO: edit QA model name here
qa_model_name = "Llama-3.1-8B"
# qa_model_name = "Llama-3.1-70B"
# qa_model_name = "o1"


file_input = "outputs/2_ans_sub_output/{}_{}_{}_{}.json".format(qa_model_name.lower(), str(number_of_shot), setting_type, data_name)
file_out_eval = "outputs/2_ans_sub_output_scores/{}_{}_{}_{}.json".format(qa_model_name.lower(), str(number_of_shot), setting_type, data_name)

data = utils.read_json(file_input)

data_out = []


# Use Llama-3.3-70B as a judge
judge_model_name = "Llama-3.3-70B"
model_id = "meta-llama/{}-Instruct".format(judge_model_name)


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
count_correct = 0

for item in tqdm(data):
    decomposition = item["decomposition"]
    for idx, subitem in enumerate(decomposition):
        subitem["pred_ans"] = utils.extract_answer(subitem["generated_ans"]).strip()
        #
        new_item = {}
        new_item["_id"] = item["_id"] + "_" + str(idx+1)
        new_item["question"] = subitem["question"]
        new_item["gold_ans"] = subitem["answer"]
        new_item["generated_ans"] = subitem["generated_ans"]
        # new_item["gold_ans_id"] = subitem["answer_id"]
        new_item["pred_ans"] = subitem["pred_ans"]
        # new_item["context"] = item["context"]
        
        #
        em, f1, prec, recall = update_answer(metrics, subitem["pred_ans"], new_item["gold_ans"])
        new_item["EM"] = em
        new_item["f1"] = f1
        if em == True:
            new_item["LLM-judge"] = "CORRECT"
        elif new_item["pred_ans"] == "":
            new_item["LLM-judge"] = "INCORRECT"
        elif new_item["pred_ans"].lower() == "I cannot confidently find the answer".lower():
            new_item["LLM-judge"] = "NOT_SURE"
        else:                
            llm = judge_pred_answer(pipeline, subitem["question"], new_item["gold_ans"], new_item["pred_ans"], item["context"])
            new_item["LLM-judge"] = utils.extract_answer(llm).strip()
        data_out.append(new_item)


utils.write_json(data_out, file_out_eval, indent=3)
