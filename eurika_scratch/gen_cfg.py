class cfg_cartpole:
    seed = 0
    task_name = "CartPole-v1"
    sim_device = "cpu"
    rl_device = "cpu"
    graphics_device_id = 0
    headless = False
    multi_gpu = False
    capture_video = False
    force_render = False
    capture_video_freq = 1
    capture_video_len = 1000
    test = False
    iteration = 10
    training_filename = "cartpole_train.py"
    task = "CartPole"
    task_name = "CartPole-v1"
    task_env = "CartPoleEnv"
    task_description = "Balance a pole on a cart by moving the cart left or right."

    model = "gpt-4"  # LLM model (other options are gpt-4, gpt-4-0613, gpt-3.5-turbo-16k-0613)
    temperature = 1.0
    suffix = "GPT"  # suffix for generated files (indicates LLM model)

    # Eureka parameters
    iteration = 1 # how many iterations of Eureka to run
    sample = 1 # number of Eureka samples to generate per iteration
    max_iterations = 3000 # RL Policy training iterations (decrease this to make the feedback loop faster)
    num_eval = 5 # number of evaluation episodes to run for the final reward
    capture_video = False # whether to capture policy rollout videos

    train_params = {
        'config': {
            'multi_gpu': False,
            'seed': 0,
            'num_actors': 1,
            'num_actors_per_gpu': 1,
            'num_gpus': 1,
            'num_cpus': 1,
            'num_envs_per_actor': 1,
            'num_envs': 1,
            'num_envs_per_gpu': 1,
            'num_envs_per_cpu': 1  # Assuming you want to set this to 1, as it was missing a value
        }
    }


"""
class cfg_humanoid:
    seed = 0
    task_name = "Humanoid-v4"
    sim_device = "cpu"
    rl_device = "cpu"
    graphics_device_id = 0
    headless = False
    multi_gpu = False
    capture_video = False
    force_render = False
    capture_video_freq = 1
    capture_video_len = 1000
    test = False
    iteration = 10
    task = "Humanoid"
    task_name = "Humanoid-v4"
    task.env = "HumanoidEnv"
    task.env.env_name = "Humanoid"
    train = {}
    train.params = {}
    train.params.config = {}
    train.params.config.multi_gpu = False
    train.params.config.seed = 0
    train.params.config.num_actors = 1
    train.params.config.num_actors_per_gpu = 1
    train.params.config.num_gpus = 1
    train.params.config.num_cpus = 1
    train.params.config.num_envs_per_actor = 1
    train.params.config.num_envs = 1
    train.params.config.num_envs_per_gpu = 1
    train.params.config.num_envs_per_cpu = 1


class cfg_reacher:
    seed = 0
    task_name = "Reacher-v4"
    sim_device = "cpu"
    rl_device = "cpu"
    graphics_device_id = 0
    headless = False
    multi_gpu = False
    capture_video = False
    force_render = False
    capture_video_freq = 1
    capture_video_len = 1000
    test = False
    iteration = 10
    task = "CartPole"
    task_name = "CartPole-v1"
    task.env = "CartPoleEnv"
    task.env.env_name = "CartPole"
    train = {}
    train.params = {}
    train.params.config = {}
    train.params.config.multi_gpu = False
    train.params.config.seed = 0
    train.params.config.num_actors = 1
    train.params.config.num_actors_per_gpu = 1
    train.params.config.num_gpus = 1
    train.params.config.num_cpus = 1
    train.params.config.num_envs_per_actor = 1
    train.params.config.num_envs = 1
    train.params.config.num_envs_per_gpu = 1
    train.params.config.num_envs_per_cpu

class cfg_pusher:
    seed = 0
    task_name = "Pusher-v4"
    sim_device = "cpu"
    rl_device = "cpu"
    graphics_device_id = 0
    headless = False
    multi_gpu = False
    capture_video = False
    force_render = False
    capture_video_freq = 1
    capture_video_len = 1000
    test = False
    iteration = 10
    task = "CartPole"
    task_name = "CartPole-v1"
    task.env = "CartPoleEnv"
    task.env.env_name = "CartPole"
    train = {}
    train.params = {}
    train.params.config = {}
    train.params.config.multi_gpu = False
    train.params.config.seed = 0
    train.params.config.num_actors = 1
    train.params.config.num_actors_per_gpu = 1
    train.params.config.num_gpus = 1
    train.params.config.num_cpus = 1
    train.params.config.num_envs_per_actor = 1
    train.params.config.num_envs = 1
    train.params.config.num_envs_per_gpu = 1
    train.params.config.num_envs_per_cpu

"""