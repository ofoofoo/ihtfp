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
from gen_cfg import cfg_cartpole, cfg_lunarlander
from gpt_parsing import extract_code, get_reward_function_from_string
# import torch

logging.basicConfig(level=logging.INFO)
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_gpt_responses(cfg, message=None):
    promptData = PromptData(cfg)
    responses = []
    response_cur = None
    total_samples = 0
    total_token = 0
    total_completion_token = 0
    chunk_size = cfg.sample

    logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

    gpt_message = promptData.messages
    if message is not None:
        gpt_message += message
        # logging.info(f"Message: {gpt_message}")
        assert len(gpt_message) == 4, "expect 2 additional messages to gpt"
    

    while True:
        if total_samples >= cfg.sample:
            break
        for attempt in range(1000):
            try:
                response_cur = openai.ChatCompletion.create(
                    model=cfg.model,
                    messages=gpt_message,
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

    # if cfg.sample == 1:
    #     logging.info(f"Input: {promptData.messages}\n")
    #     logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")
    
    return responses

def run_for_response(cfg, code_string, prev_model=None, index = -1):
    """
    Run the code for the given response
    
    Returns:
    - 
    - 
    """
    # logging.info(f"Code String: \n\n{code_string}")

    reward_func = get_reward_function_from_string(code_string)

    if reward_func is None:
        logging.info("Failed to extract reward function from GPT output")
        return
    
    if cfg.task == "CartPole":
        logging.info("Running CartPole")
        from envs.cartpole_train import train_cartpole
        model, fitness_values, avg_rewards = train_cartpole(cfg, reward_func, prev_model, index)
        fitness_values = fitness_values[:750]
        avg_rewards = fitness_values[:750]
        logging.info(f"Done running Cartpole for index {index}")
        #logging.info(f"Average fitness value across training episodes: {fitness_values} and number of episodes: {len(fitness_values)}")
        return model, fitness_values, avg_rewards
        
    
    elif cfg.task == "LunarLander":
        logging.info("Running LunarLander")
        from envs.lunarlander_train import train_lunarlander
        model = train_lunarlander(reward_func, None)
        logging.info("Done running LunarLander")
        logging.info(f"Got model: {model}")


model_runs = []


class ModelInfo:
    def __init__(self, model_from, generation):
        self.model_from = model_from
        self.index = len(model_runs)
        self.done_training = False
        self.generation = generation
        if self.generation > 1:
            assert self.model_from is not None, "Model from cannot be None for generation > 1"
        self.avg_rewards = [-1000000000]
        self.fitness_values = [-1000000000]
        self.model = None
    
    def train(self, cfg, code_string):
        logging.info(f"Training model for generation {self.generation} and index {self.index}")
        self.code_string = code_string
        if self.done_training:
            logging.warn("Model already trained")
            return
        if cfg.task == "CartPole":
            try:
                model, fitness_values, avg_rewards = run_for_response(cfg, code_string, self.model_from, self.index)
            except Exception as e:
                model = None
                fitness_values = [-1000000000] * 100
                avg_rewards = [-1000000000] * 100
                return

            self.model = model
            self.output_info = (f"Model generation {self.generation}: trained with fitness values: {fitness_values} and average rewards: {avg_rewards}\n")
            self.fitness_values = fitness_values
            self.avg_rewards = avg_rewards

            with open("output.txt", "a") as file:
                file.write(f"model_{self.index}: {model_runs[-1].fitness_values[-1]}\n")

            promptData = PromptData(cfg)
            self.output_info += promptData.policy_feedback
            
            logging.info(self.output_info)
            self.done_training = True
        else:
            assert(False, "Task not supported")
    
    def spawn_next_gen(self, cfg, generation):
        if self.generation == 0:
            self.done_training = True
        
        if not self.done_training:
            logging.warn("Model not trained yet, but trying to spawn child tasks")
            return

        messages = []
        if self.generation > 0:
            messages += [{"role": "assistant", "content": self.code_string}]
            messages += [{"role": "user", "content": self.output_info}]
        
        if len(messages) == 0:
            messages = None
        
        responses = get_gpt_responses(cfg, messages)
        for response_id in range(len(responses)):
            logging.info(f"Iteration {generation}: Processing Code Run {response_id}")
            response_content = responses[response_id]["message"]["content"]
            code_string = extract_code(response_content)
            model_runs.append(ModelInfo(self.model, self.generation + 1))
            model_runs[-1].train(cfg, code_string)
    
    
    def __str__(self):
        return f"Model: {self.model}, Index: {self.index}"


def models_of_generations(generation):
    """
    Returns the models of the given generation, or list of generations
    """
    return [model for model in model_runs if model.generation == generation or model.generation in generation]

def top_k_models(generation = None, k = 1):
    """
    Returns the best model of the given generation
    """
    models = models_of_generations(generation if generation is not None else range(0, len(model_runs)))
    return sorted(models, key=lambda x: x.fitness_values[-1], reverse=True)[:k]


def main(cfg):
    # assert(os.getcwd() == EURIKA_ROOT_DIR)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Your code logic here
    # Read the cfg file and perform necessary operations

    model_runs.append(ModelInfo(None, 0))
    
    for generation in range(1, cfg.iteration+1):

        logging.info(f"Generation {generation}, taking top {cfg.top_k} models from previous generation")
        best_models = top_k_models(generation - 1, cfg.top_k)
        for model in best_models:
            model.spawn_next_gen(cfg, generation)

    

def string_to_function():
    custom_function = """
def reward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
    """
    return get_reward_function_from_string(custom_function)

if __name__ == "__main__":
    config_dict = {
            'CartPole-v1': cfg_cartpole,
            'LunarLander-v3': cfg_lunarlander
        }
    task_name = "CartPole-v1"
    cfg = config_dict.get(task_name)
    main(cfg)