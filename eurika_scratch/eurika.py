import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
from pathlib import Path
from prompts import PromptData
import shutil
import time 
from constants import EUREKA_ROOT_DIR, EURIKA_ROOT_DIR
from gen_cfg import cfg_cartpole
from gpt_parsing import extract_code, get_reward_function_from_string
import torch

logging.basicConfig(level=logging.INFO)
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_gpt_responses(cfg):
    promptData = PromptData(cfg)
    responses = []
    response_cur = None
    total_samples = 0
    total_token = 0
    total_completion_token = 0
    chunk_size = cfg.sample if "gpt-3.5" in cfg.model else 4

    logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

    while True:
        if total_samples >= cfg.sample:
            break
        for attempt in range(1000):
            try:
                response_cur = openai.ChatCompletion.create(
                    model=cfg.model,
                    messages=promptData.messages,
                    temperature=cfg.temperature,
                    n=chunk_size
                )
                total_samples += chunk_size
                break
            except Exception as e:
                if attempt >= 10:
                    chunk_size = max(int(chunk_size / 2), 1)
                    print("Current Chunk Size", chunk_size)
                logging.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()

        responses.extend(response_cur["choices"])
        prompt_tokens = response_cur["usage"]["prompt_tokens"]
        total_completion_token += response_cur["usage"]["completion_tokens"]
        total_token += response_cur["usage"]["total_tokens"]

    if cfg.sample == 1:
        logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")
    
    return responses

def run_for_response(cfg, response_content):
    """
    Run the code for the given response
    
    Returns:
    - 
    - 
    """
    promptData = PromptData(cfg)
    code_string = extract_code(response_content)
    logging.info(f"Code String: \n\n{code_string}")

    reward_func = get_reward_function_from_string(code_string)

    if reward_func is None:
        logging.info("Failed to extract reward function from GPT output")
        return
    
    if cfg.task == "CartPole":
        logging.info("Running CartPole")
        from envs.cartpole_train import train_cartpole
        model = train_cartpole(reward_func, None)
        logging.info("Done running Cartpole")
        logging.info(f"Got model: {model}")
    
    else if cfg.task == "LunarLander":
        logging.info("Running LunarLander")
        from envs.lunarlander_train import train_lunarlander
        model = train_lunarlander(reward_func, None)
        logging.info("Done running LunarLander")
        logging.info(f"Got model: {model}")

    





def main(cfg):
    # assert(os.getcwd() == EURIKA_ROOT_DIR)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Your code logic here
    # Read the cfg file and perform necessary operations
    
    for iter in range(cfg.iteration):
        responses = get_gpt_responses(cfg)

        for response_id in range(len(responses)):
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")
            response_content = responses[response_id]["message"]["content"]
            run_for_response(cfg, response_content)


        pass
    

def string_to_function():
    custom_function = """
def reward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
    """
    return get_reward_function_from_string(custom_function)

if __name__ == "__main__":
    config_dict = {
            'CartPole-v1': cfg_cartpole,
            'LunarLander-v2': cfg_lunarlander
        }
    task_name = "CartPole-v1"
    cfg = config_dict.get(task_name)
    main(cfg)