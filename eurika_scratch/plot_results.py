import matplotlib.pyplot as plt


# def get_result_as_dict(filename):
#     result = {}
#     with open(filename, 'r') as f:
#         for line in f:
#             print('asga')
#             print(line.strip().split(':'))
#             key, value = line.strip().split(':')
#             result[key] = float(value)
#     return result

import json

def get_result_as_dict(filename):
    with open(filename, 'r') as file:
        result = json.load(file)
    return result


num_models = 4
all_results = [get_result_as_dict(f'./model_{i}.txt') for i in range(1, num_models + 1)]
# print(all_results)

def plot_generation_vs_eval_hacky(generations, model_evals, keyword):
    plt.figure()
    plt.plot(generations, model_evals)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Across Generations')
    plt.savefig(f'eval_plot_{keyword}.png') 

"""
def plot_generation_vs_fitness():
    plt.figure()
    for i in range(num_models):
        plt.plot(all_results[i]['generation'], all_results[i]['fitness'], label=f'Model {i+1}')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

def plot_generation_vs_eval_fitness():
    plt.figure()
    for i in range(num_models):
        plt.plot(all_results[i]['generation'], all_results[i]['fitness'], label=f'Model {i+1}')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

plot_generation_vs_fitness()
"""
