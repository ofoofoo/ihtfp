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
    if code_string:
        task_code_string = code_string
        namespace = {}
        code_string = "import numpy as np\nimport math \n\n" + code_string
        
        try:
            exec(code_string, namespace)
            custom_function = namespace['compute_reward']
            return custom_function
        except:
            print("Bad code input")
            return None
            
    else:
        return None