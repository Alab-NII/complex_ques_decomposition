This is the GitHub repo for the paper: [Identifying Where Large Language Models Struggle in Answering
Complex Questions]()

## Reproduction of Results

- download [data]()
- download [outputs]()

### Table 1: Automatic and human scores (green) in the decomposition stage
```bash
python3 eval_decompose_automatic.py
```

```bash
python3 eval_decompose_human.py
```

### Table 2: LLM-as-a-Judge accuracy (based on Llama 3.370B) in the sub-problem-solving stage

```bash
python3 eval_sub_problem_get_scores.py
```

### Table 3: LLM-as-a-Judge accuracy (Llama 3.3 70B) for full-QA performance using zero-shot-CoT

```bash
python3 eval_full_get_scores.py
```


## Running process

### Stage 1: Decomposition
```bash
python3 run_s1_decompose.py
```


### Stage 2: Subproblem Solving
```bash
python3 run_s2_ans_sub.py
```


#### Stage 2: Evaluation 
```bash
python3 eval_sub_problem_llm.py
```


### Full-QA  
```bash
python3 run_full_qa.py
```


#### Full-QA: Evaluation 
```bash
python3 eval_full_qa_llm.py
```