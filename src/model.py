from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src import utils as ut
import os, json
import torch


class model:
    def __init__(self, model_dir : str, model_name : str, num_gpus : int = 0):
        self.path = model_dir
        self.name = model_name
        self.num_gpus = torch.cuda.device_count()
        self.gpus = range(self.num_gpus)
        if num_gpus > 0:
            self.num_gpus = num_gpus

    def serial_inference_chat(self, input, template = None, b_size = 16):
        if template:
            with open(template, 'r') as f:
                data = json.load(f)
                system = data["system"]
        else:
            system = "You're a helpful agent that answers questions strictly based on the examples provided."
        torch.cuda.empty_cache()
        boi = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype = torch.float16, low_cpu_mem_usage = True)
        toke = AutoTokenizer.from_pretrained(self.path, torch_dtype = torch.float16)
        pipe = pipeline("text-generation", model = boi, tokenizer = toke, device=0, batch_size = b_size)
        #pipe = pipeline("text-generation", model=model)
        response = []
        if isinstance(input, list):
            for boi in input:
                '''
                messages = [{"role":"system", "content":system}]
                messages.append({"role":"user", "content":boi})
                '''
                messages = [{"role":"user", "content":boi}]
                #print(messages)
                generated = pipe(
                    messages,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=toke.eos_token_id,
                    max_length=2048,
                )
                messages = generated[0]["generated_text"]
                response.append(messages[-1]["content"])
                ut.log(messages, self.name)
            ut.log(response, self.name + '_response')
        elif isinstance(input, str):
            '''
            messages = [{"role":"system", "content":system}, {"role":"user", "content":input}]
            generated = pipe(messages)
            messages = generated[0]["generated_text"]
            ut.log(messages, self.name)
            '''
            print("Please input using list.")
        #ut.log(messages, "inference")
        return response

'''
input = ut.generate_input_list('assets/input.json', 'assets/input_test.json')
serial_inference(input, "../models/llama-2-13b-chat-hf", "assets/template.json")
#serial_inference(input, "../models/llama-2-13b-chat-hf")
'''