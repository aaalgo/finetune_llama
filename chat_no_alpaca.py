#!/usr/bin/env python3
import llamahf
import os
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

# # to save memory use bfloat16
torch.set_default_dtype(torch.bfloat16)

MODEL = 'decapoda-research/llama-7b-hf'
#MODEL = 'decapoda-research/llama-13b-hf'
#MODEL = 'decapoda-research/llama-30b-hf'
#MODEL = 'decapoda-research/llama-65b-hf'


tokenizer = llamahf.LLaMATokenizer.from_pretrained(MODEL)
model = llamahf.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map='auto')
print("Setup PEFT")


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops=[]):
        print('-' * 40)
        print(tokenizer.decode(input_ids[0]))
        if input_ids[0][-1] == 13:
            return True

        return False


#INIT = """A dialog, where User interacts with AI. AI tries to behave like a human being.
#User: Hello, AI.
#AI: Hello! How can I assist you today?
#"""

#ctx = INIT

INIT = """A dialog, where User interacts with AI. AI tries to behave like a human being.
User: Hello, AI.
AI: Hello! How can I assist you today?
User:
"""

while True:
    #print('-' * 40)
    #print(ctx.rstrip("\n"))
    prompt = input(f'User: ')
    ctx = INIT.strip() + ' ' + prompt.strip() + '\n'

    if len(ctx.strip()) > 0:
        batch = tokenizer(ctx, return_tensors="pt")
        result = model.generate(input_ids=batch["input_ids"].to(model.device),
                                do_sample=True,
                                top_k=50,
                                max_length=2048,
                                top_p=0.95,
                                temperature=1.0,
                                #temperature=0.5,
                                stopping_criteria=StoppingCriteriaList([StoppingCriteriaSub()])
                                # repetition_penalty=1.17
                                )
        decoded = tokenizer.decode(result[0])
        print(decoded)
        print()
