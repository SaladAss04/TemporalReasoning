import json, os, csv
import random as rd
import numpy as np
import pandas as pd
from typing import Literal
from itertools import product, cycle

def fill_template(template_str, values_dict):
    """
    填充字符串模板的函数。

    参数:
    template_str (str): 字符串模板，包含键的占位符（如 {key}）。
    values_dict (dict): 包含键值对的字典，用于填充模板中的占位符。

    返回:
    str: 填充好的字符串。
    """
    try:
        return template_str.format(**values_dict)
    except KeyError as e:
        return f"Error: Missing value for {e.args[0]}"

def generate_mapping(A, B):
    """
    Generate a surjection from A to B. Requires card(A) >= card(B).
    input: list
    return: [tuple(i, j)]
    """
    mapping = {}
    for b in B:
        # 随机选择一个A中的元素
        a = A.pop(rd.randint(0, len(A) - 1))
        mapping[a] = b
    # Step 2: 剩余的A元素随机映射到B
    for a in A:
        # 随机选择一个B中的元素
        b = rd.choice(B)
        mapping[a] = b
    return mapping
        
def find_by_first_column(csv_file, value):
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == value:  # 比较第一列的值
                return row  # 返回找到的行
    return None        

def generate_answer_param(q_param, biodata):
    """
    Currently 4 types.
    """
    answer = {}
    if "t1" in q_param.keys() and "t2" in q_param.keys():
        action = []
        for i in range(3):
            t = "t" + str(i+1)
            if biodata[t] >= q_param["t1"] and biodata[t] <= q_param["t2"]:
                action.append(biodata["e" + str(i+1)])
        if len(action) >= 1:
            answer["period"] = ','.join(action)
        else:
            answer["period"] = "None"
    if "t3" in q_param.keys():
        answer["stamp"] = "None"
        for i in range(3):
            t = "t" + str(i+1)
            if biodata[t] == q_param["t3"]:
                answer["stamp"] = biodata["e" + str(i+1)]
                break
    if "a" in q_param.keys():
        answer["age"] = "None"
        for i in range(3):
            t = "t" + str(i+1)
            if int(biodata[t]) - int(biodata["t0"]) == int(q_param["a"]):
                answer["age"] = biodata["e" + str(i+1)]
                break
    if "x" in q_param.keys() and "y" in q_param.keys():
        time0 = biodata["t1"]
        answer["years"] = "None"
        for i in range(3):
            t = "t" + str(i+1)
            if int(biodata[t]) - int(time0) == int(q_param["x"]):
                answer["years"] = biodata["e" + str(i+1)]
                break        
    return answer
        
class question_set:
    def __init__(self, dir: str, biodata: str = None, template: Literal["llama-2"] = None, question_limit: Literal["None", "OnlyEasy", "OnlyHard"] = "None", eg_num = 3) -> None:
        with open(dir, 'r') as f:
            data = json.load(f)
        self.q_type = data["question_template"]
        self.q_type_code = ['period', 'stamp', 'age', 'years']
        if question_limit == "OnlyEasy":
            self.q_type = self.q_type[:2]
            self.q_type_code = self.q_type_code[:2]
        elif question_limit == "OnlyHard":
            self.q_type = self.q_type[2:4]
            self.q_type_code = self.q_type_code[2:4]
        self.b_type = data["biodata_template"]
        #Both are lists of string templates
        if biodata:
            self.b_path = biodata
        else:
            self.b_path = None
        self.input_path = None
        self.template = template
        self.eg_num = eg_num
        
    def update(self, dir: dict = None, biodata: str = None, template: Literal["llama-2"] = None) -> None:
        if dir:
            self.q_type = dir["q"]
            self.b_type = dir["b"]
        if biodata:
            self.b_path = biodata
        if template:
            self.template = template
        
    def generate_question(self, code: str, exemplar: Literal['None', 'Surjection']):
        """
        generates questions alongside answer, covering all entries in self.b_data.
        returns the list of dicts containing q-b-a, and stores them in self.input_path.
        """
        mapping = []
        if exemplar == 'None':
            mapping = list(product(self.q_type, self.b_type))
        elif exemplar == 'Surjection':
            mapping = generate_mapping(self.q_type, self.b_type)
            mapping = mapping.items()
        assert len(mapping) > 0
        cycler = cycle(mapping)
        #print(mapping)

        ret = []
        with open(self.b_path, 'r') as f:
            csv_reader = csv.DictReader(f)
            dict_list = [row for row in csv_reader]
        
        for bio in dict_list:
            boi = next(cycler)
            entry = {"id":bio["id"]}
            question_param = {
                "n": bio["n"],
                "t1": str(int(bio["t2"]) + rd.randint(-10, 10)),
                "t2": str(int(bio["t3"]) + rd.randint(-10, 10)),
                "t3": str(int(bio["t3"]) + rd.randint(-1, 2)),
                "a": str(int(bio["t3"]) + rd.randint(-1, 2) - int(bio["t0"])),
                "x": str(int(bio["t2"]) - int(bio["t1"]) + rd.randint(-2, 2)),
                "y": bio["e1"]
            }
            answers = generate_answer_param(question_param, bio)
            question_text, bio_text = fill_template(boi[0], question_param), fill_template(boi[1], bio)
            if "done between" in question_text:
                answer_text = answers["period"]
                entry["q_type"] = "period"
            elif "do in" in question_text:
                answer_text = answers["stamp"]
                entry["q_type"] = "stamp"
            elif "do at the age of" in question_text:
                answer_text = answers["age"]    
                entry["q_type"] = "age"
            elif "years after" in question_text:
                answer_text = answers["years"] 
                entry["q_type"] = "years"
            else:
                print(question_text)
                raise ValueError("Can't match answer.")  
            entry['q'], entry['b'], entry['a'] = question_text, bio_text, answer_text
            ret.append(entry)
        with open('assets/' + code + '.json', 'w') as f:
            json.dump(ret, f)
            self.input_path = 'assets/' + code + '.json'
        return ret

    def prepare_input(self, qtype : str = Literal['period', 'stamp', 'age', 'years'], entry_num = None):
        """
        from the corpus under self.input_path, performs example/test split then organizes a list of strings for query.
        Returns the string, doesn't save.
        """
        with open(self.input_path, 'r') as f:
            data = json.load(f)
            related = [x for x in data if x['q_type'] == qtype]
        num = min(self.eg_num, len(related) - 1)
        chosen_examples = rd.sample(related, num)
        with open('assets/template.json', 'r') as f:
            data = json.load(f)
            template = data["template"]
            inst = data["instruction"]
        example_text = fill_template(template[0], {"i":inst}) + "###Examples###:\n"
        input_text = []
        input_answer = []
        input_id = []
        for x in chosen_examples:
            example_text += fill_template(template[1], x)
        if entry_num:
            e_num = min(entry_num, len(related) - num)
        else:
            e_num = len(related) - num
        chosen_questions = rd.sample([y for y in related if y not in chosen_examples], e_num)
        for x in chosen_questions:
            input_text.append(example_text + fill_template(template[2], x))
            input_answer.append(x['a'])
            input_id.append(x['id'])
        assert len(input_text) >= 1
        return [(input_text[i], input_answer[i], input_id[i]) for i in range(len(input_text))]
    
    def prepare_all_input(self, entry_num = None):
        ret = {}
        for t in self.q_type_code:
            ret[t] = self.prepare_input(t, entry_num)
        #print("each type: ", len(ret["period"]), len(ret["stamp"]), len(ret["age"]), len(ret["years"]))
        ret = [item for sublist in ret.values() for item in sublist]
        with open('log/input_text.json', 'w') as f:
            json.dump(ret, f)
        df = pd.DataFrame({'input': [x[0] for x in ret], 'answer': [x[1] for x in ret], 'id': [x[2] for x in ret]})
        return df

    def semantic_match(self, input: str, answer: str, id: str):
        x = input.split("Answer:")[-1].lower()
        x.strip('*')
        x.strip('#')
        answer_events = [x.strip().lower() for x in answer.split(',')]
        for z in answer_events:
            if z not in x:
                return 'W'
        line = find_by_first_column(self.b_path, id)
        all_events = [line[4], line[6], line[8]]
        rest_events = [y for y in all_events if y not in answer_events]
        if len(rest_events) > 0:
            for z in rest_events:
                if z in x:
                    return 'W'
        return 'R'
        
    def evaluate(self, response, df):
        correct = 0
        for x, y in zip(response, df.itertuples()):
            result = self.semantic_match(x, y.answer, y.id)
            if result == 'R':
                correct += 1
                #log("correct response: " + x + " ground truth:, " + y[1], "response_legend")
        print("correct answers: ", correct, "out of: ", len(response))
        return correct / len(response)
        
def log(input, head, format = "json"):
    if format == "json":
        if isinstance(input, list):
            if not os.path.exists('log/' + head + '.json'):
                with open('log/' + head + '.json', 'w') as f:
                    f.write('[]')
            with open('log/' + head + '.json', 'r') as f:
                try:
                    data = json.load(f)
                except:
                    print("Unable to concatenate.")
                    data = []
            data.extend(input)
            with open('log/' + head + '.json', 'w') as f:
                json.dump(data, f)
        else:
            print("Please provide a list to register")
    elif format == "txt":
        with open('log/' + head + '.txt', 'a') as f:
             f.write("\n\n" + input)


    
    
#generate_example('assets/test.csv', 'assets/template.json', 'assets/input_test.json')