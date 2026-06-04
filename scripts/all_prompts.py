import sys 


system_promt = f"""
		You are an expert in question answering systems.
		"""
		

def get_prompt(question, context):
	prompt = f"""Break down the following complex question into a series of simpler sub-questions that follow a logical progression.
			### 
			* Ensure each sub-question builds upon the previous ones, using placeholders (e.g., #Ans1 for the first answer) to illustrate dependencies.
			* If decomposition is difficult, you may refer to the provided context {context}, but do not directly use words from the context or introduce information not present in the original question.
			* Avoid unnecessary decomposition or trivial sub-questions.
			* Your response will be evaluated based on clarity, conciseness, and correctness.
			* Ensure that each sub-question has a single, well-defined answer.
			### 
			Format your output as follows to maintain consistency:
			Sub-question 1: [First sub-question]  
			Sub-question 2: [Second sub-question, incorporating #Ans1 if applicable]  
			Sub-question 3: [Third sub-question, incorporating #Ans2 if applicable]  
			### 
			Complex question: {question}
		"""
	return prompt


def generate_subques(pipeline, question, context):
	prompt = get_prompt(question, context)
	messages = [
		{"role": "system", "content": system_promt},
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
	return prompt, answer


def obtain_format_subques(cur_decompositions):
	decompositions = []
	for item in cur_decompositions:
		decompositions.append(item["question"])
	# Generate the formatted string with dynamic numbering
	formatted_decompositions = "\n".join([f"Sub-question {i+1}: {decompositions[i]}" for i in range(len(decompositions))])
	return formatted_decompositions


def combine_shots(examples):
	prompt = ""
	#
	for i, item in enumerate(examples, start=1):
		subques = obtain_format_subques(item["decomposition"])
		prompt += f"#### Example {i}:\n"
		prompt += f"* Complex question: {item['question']}\n"
		prompt += f"* Context: {item['context']}\n"
		prompt += f"* Decomposed sub-questions: {subques}\n\n"
	
	return prompt.strip()


def get_prompt_fewshot(question, context, examples):
	demonstrations = combine_shots(examples)
	#
	prompt = f"""Break down the following complex question into a series of simpler sub-questions that follow a logical progression.
			### 
			* Ensure each sub-question builds upon the previous ones, using placeholders (e.g., #Ans1 for the first answer) to illustrate dependencies.
			* If decomposition is difficult, you may refer to the provided context {context}, but do not directly use words from the context or introduce information not present in the original question.
			* Avoid unnecessary decomposition or trivial sub-questions.
			* Your response will be evaluated based on clarity, conciseness, and correctness.
			* Ensure that each sub-question has a single, well-defined answer.
			### 
			Format your output as follows to maintain consistency:
			Sub-question 1: [First sub-question]  
			Sub-question 2: [Second sub-question, incorporating #Ans1 if applicable]  
			Sub-question 3: [Third sub-question, incorporating #Ans2 if applicable]  
			### 
			Complex question: {question}
			###
			Here are {len(examples)} reference examples:
			{demonstrations}
		"""
	return prompt

 
def generate_subques_fewshot(pipeline, question, context, examples):
	prompt = get_prompt_fewshot(question, context, examples)
	messages = [
		{"role": "system", "content": system_promt},
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
	return prompt, answer


# For the sub-question answering stage
def get_prompt_answer_subques_zeroshot(question, context):
	#
	prompt = f"""Answer the following question based on the provided context.
            ### Instructions:
            * Context: {context}
            * Your task is to identify the correct answer from the provided context.
            * If you are not confident in your answer, return: <ans> I cannot confidently find the answer </ans>.
            ### Response Format:
            Please format your answer within brackets as follows: <ans> Your Answer </ans>.
            ### 
            Question: {question}
		"""
	return prompt


def answer_subques_zeroshot(pipeline, question, context):
	prompt = get_prompt_answer_subques_zeroshot(question, context)
	messages = [
		{"role": "system", "content": system_promt},
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
	return prompt, answer


def get_prompt_answer_ques_cot(question, context):
	#
	prompt = f"""Answer the following question based on the provided context using chain-of-thought reasoning.
            ### Instructions:
            * Context: {context}
            * Your task is to identify the correct answer from the provided context.
            * You can break down the problem into smaller parts or think step-by-step and reason through your answer.
            * If you are not confident in your answer, return: <ans> I cannot confidently find the answer </ans>.
            ### Response Format:
            Please format your answer within brackets as follows: <ans> Your Answer </ans>.
            ### 
            Question: {question}
		"""
	return prompt


def answer_ques_cot(pipeline, question, context):
	prompt = get_prompt_answer_ques_cot(question, context)
	messages = [
		{"role": "system", "content": system_promt},
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
	return prompt, answer