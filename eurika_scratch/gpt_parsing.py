from utils.extract_task_code import *
import logging

def extract_code(gpt_str):
    patterns = [
        r'```python(.*?)```',
        r'```(.*?)```',
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    for pattern in patterns:
        code_string = re.search(pattern, gpt_str, re.DOTALL)
        if code_string is not None:
            code_string = code_string.group(1).strip()
            break
    code_string = gpt_str if not code_string else code_string
    return code_string

def clean_imports(gpt_str):
    lines = code_string.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            code_string = "\n".join(lines[i:])

def get_code_string(gpt_str):
    code_string = extract_code(gpt_str)
    code_string = clean_imports(code_string)
    return code_string

def get_reward_function_from_string(code_string):
    """
    Returns the complete environment code from a gpt output, or None if failed
    """
    
    task_code_string = code_string
    namespace = {}
    code_string = "import torch\n \n" + code_string
    print(code_string)
    exec(code_string, namespace)
    custom_function = namespace['compute_reward']

    return custom_function

    # Add the Eureka Reward Signature to the environment code
    # try:
    #     gpt_reward_signature, input_lst = get_function_signature(code_string)
    # except Exception as e:
    #     logging.info(f"Error: cannot parse function signature!")
    #     return None

    # reward_signature = [
    #     f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
    #     f"self.extras['gpt_reward'] = self.rew_buf.mean()",
    #     f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
    # ]
    # indent = " " * 8
    # reward_signature = "\n".join([indent + line for line in reward_signature])
    # if "def compute_reward(self)" in task_code_string:
    #     task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
    # elif "def compute_reward(self, actions)" in task_code_string:
    #     task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
    # else:
    #     raise NotImplementedError

    # # Save the new environment code when the output contains valid code string!
    # with open(output_file, 'w') as file:
    #     file.writelines(task_code_string_iter + '\n')
    #     file.writelines("from typing import Tuple, Dict" + '\n')
    #     file.writelines("import math" + '\n')
    #     file.writelines("import torch" + '\n')
    #     file.writelines("from torch import Tensor" + '\n')
    #     if "@torch.jit.script" not in code_string:
    #         code_string = "@torch.jit.script\n" + code_string
    #     file.writelines(code_string + '\n')

    # with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
    #     file.writelines(code_string + '\n')
    