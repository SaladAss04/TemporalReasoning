#4 types of questions, 100 biodata entries, 3 models
from src.utils import question_set, log
from src.model import model
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

question = question_set('assets/template.json', 'assets/example.csv', question_limit = "OnlyHard")
subject = model('../models/llama-2-7b-chat-hf', 'llama-2-7b-chat-hf')

question.generate_question("initial", "None")
input_dataset = question.prepare_all_input()
'''
for x in input_dict.values():
    subject.serial_inference_chat(x, 'assets/template.json')
'''

response = subject.serial_inference_chat(input_dataset['input'].tolist())

rate = question.evaluate(response, input_dataset)

log_text = "model: " + subject.name + "\nquestion_type: " + ','.join(question.q_type_code) + "\neg_num: " + str(question.eg_num) + "\ncorrectness: " + str(rate)
print(rate)
log(log_text, 'legend_' + subject.name, 'txt')