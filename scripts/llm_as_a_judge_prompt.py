import random
from tqdm import tqdm
random.seed(42)


SYSTEM_PROMPT = """
You are an expert in evaluating the task of complex question decomposition.
"""


def get_prompt(complex_ques, decomposition, score_array_format):
	prompt = f"""You are tasked with evaluating the decomposition of a reasoning process. 
            Please assume the role of a judge and assess the decomposition based on the following criteria:   
            ## Evaluation Criteria:
            ### 1. Perfect Decomposition (1 or 0):
            * Score 1 if:
                ** All sub-questions are interconnected, forming a coherent reasoning chain.
                ** The final sub-question directly leads to the overall answer.
                ** There are no redundant or unnecessary sub-questions.
            * Score 0 if any of the above conditions are not met.
            * Note: If you assign a score of 1, you do not need to evaluate the remaining criteria for this sample.
            ### 2. Reasoning Chain (1 or 0):
            * Score 1 if:
                ** Sub-questions are connected, forming a logical path.
                ** The final sub-question leads to the overall answer, even if the reasoning is imperfect or includes redundancy.
            * Score 0 if the reasoning chain is incomplete or disconnected.                        
            ### 3. Correct Sub-Questions (Binary Evaluation: 1 or 0 for an Array): 
            * You will receive an array initialized with 0s, where the length matches the number of generated sub-questions.
            * For each sub-question that is genuinely useful in arriving at the final answer, update its corresponding position in the array to 1.
            * If a sub-question is irrelevant or redundant, leave its value as 0.
            ## Instructions:
            * Evaluate the decomposition based on the criteria above.
            * Clearly indicate the scores for each criterion in the following format:
                ** Perfect Decomposition: Your perfect decomposition score
                ** Reasoning Chain: Your reasoning chain score
                ** Correct Sub-Questions: Your score
            ## Examples:
            ### Example 1: 
            * Complex question: The Church of the Guanche People was founded in the city that is on the most populated island of what larger area?
            * Generated sub-questions:
                ** Sub-question-1: In which city was this religious organization founded?
                ** Sub-question-2: On which island is #Ans1 located?
                ** Sub-question-3: #Ans2 is the most populated island of what larger area?
            * Scores:
                ** Perfect Decomposition: 0
                ** Reasoning Chain: 1
                ** Correct Sub-Questions (Accuracy): [0, 1, 1]    
            ### Example 2: 
            * Complex question: Can telescopes hear noise?
            * Generated sub-questions:
                ** Sub-question-1: What kind of signals are these instruments primarily designed to detect?
                ** Sub-question-2: Do these signals (#Ans1) correspond to mechanical vibrations that produce sound?
                ** Sub-question-3: Given #Ans2, can these instruments pick up typical audible noise?
            * Scores:
                ** Perfect Decomposition: 0
                ** Reasoning Chain: 1
                ** Correct Sub-Questions (Accuracy): [0, 1, 0]        
                ** Correct Sub-Questions (Accuracy): [0, 1, 1]    
            ### Example 3: 
            * Complex question: What is the date of death of paternal grandfather of Alexios Komnenos?
            * Generated sub-questions:
                ** Sub-question-1: Who is the father of Alexios Komnenos?
                ** Sub-question-2: Who is the father of #Ans1?
                ** Sub-question-3: What is the date of death of #Ans2?
            * Scores:
                ** Perfect Decomposition: 1
                ** Reasoning Chain: No need to proceed since the perfect decomposition score is 1.
                ** Correct Sub-Questions (Accuracy): No need to proceed since the perfect decomposition score is 1.
            ## Here is your task:
            * Complex question: {complex_ques}
            * Generated sub-questions: {decomposition}     
            * Please put your scores here:
                ** Perfect Decomposition: Your perfect decomposition score
                ** Reasoning Chain: Your reasoning chain score
                ** Correct Sub-Questions: Your score for {score_array_format}       
		"""
	#
	return prompt


def judge_answer(pipeline, complex_ques, decomposition, score_array_format):
    prompt = get_prompt(complex_ques, decomposition, score_array_format)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_ids = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True)
    # 
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    # 
    outputs = pipeline(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=False)
    answer = outputs[0]["generated_text"][len(input_ids):]
    return answer 


SYSTEM_PROMPT_ANS = f"""
		You are an expert in question answering systems.
		"""


# For QA evaluation
def get_prompt_judge_pred_ans(ques, gold_ans, pred_ans, context):
	prompt = f"""Your job is to evaluate a predicted answer by comparing it against the gold answer and the given question. 
                You may refer to the provided context if needed.
            ## Grading Criteria:
            * "CORRECT": The predicted answer matches the gold answer or is a valid alternative (e.g., different but correct ways of writing a name).
            * "INCORRECT": The predicted answer is clearly wrong or does not align with the gold answer.
            * "NOT_SURE": It is unclear whether the predicted answer is correct based on the available information.
            ## Response Format:
            Please format your answer within brackets as follows: <ans> Your Answer </ans>.
            ## Here are some examples:
            ### Example 1:
            * Question: Charlie Rowe plays Billy Costa in which film?
            * Gold Answer: The Golden Compass (film)
            * Predicted Answer: The Golden Compass
            * Label: <ans> CORRECT </ans>
            ### Example 2:
            * Question: which King made the 1925 Birthday Honours?
            * Gold Answer: George V
            * Predicted Answer: King George V
            * Label: <ans> CORRECT </ans>
            ### Example 3:
            * Question: Ronaldo is responsible for popularized a soccer move first reported to be used by what player?
            * Gold Answer: Law Adam
            * Predicted Answer: Pedro Calomino
            * Label: <ans> INCORRECT </ans>
            ## Here is your task:
            * Question: {ques}
            * Gold Answer: {gold_ans}
            * Predicted Answer: {pred_ans}
            * Context: {context}
            * <ans> YOUR LABEL </ans>
		"""
	#
	return prompt
    
            
    
def judge_pred_answer(pipeline, complex_ques, gold_ans, pred_ans, context):
    prompt = get_prompt_judge_pred_ans(complex_ques, gold_ans, pred_ans, context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ANS},
        {"role": "user", "content": prompt},
    ]
    input_ids = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True)
    # 
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    # 
    outputs = pipeline(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=False)
    answer = outputs[0]["generated_text"][len(input_ids):]
    return answer 