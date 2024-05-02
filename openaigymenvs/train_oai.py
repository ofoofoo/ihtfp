# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import logging
import os
import datetime

import isaacgym

import hydra
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
import gym
import sys 
import shutil
from pathlib import Path

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

# ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    # print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    rank = int(os.getenv("LOCAL_RANK", "0"))
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg.train.params.config.multi_gpu = cfg.multi_gpu

    # Save the environment code!
    try:
        output_file = f"{ROOT_DIR}/tasks/{cfg.task.env.env_name.lower()}.py"
        shutil.copy(output_file, f"env.py")
    except:
        import re
        def camel_to_snake(name):
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        output_file = f"{ROOT_DIR}/tasks/{camel_to_snake(cfg.task.name)}.py"

        shutil.copy(output_file, f"env.py")

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    observers = [RLGPUAlgoObserver()]

    if cfg.wandb_activate and rank ==0 :

        import wandb
        
        # initialize wandb only once per horovod run (or always for non-horovod runs)
        wandb_observer = WandbAlgoObserver(cfg)
        observers.append(wandb_observer)

    # dump config dict
    exp_date = cfg.train.params.config.name + '-{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    experiment_dir = os.path.join('runs', exp_date)
    print("Network Directory:", Path.cwd() / experiment_dir / "nn")
    print("Tensorboard Directory:", Path.cwd() / experiment_dir / "summaries")

    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    rlg_config_dict['params']['config']['log_dir'] = exp_date

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    statistics = runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint' : cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })

    if cfg.wandb_activate and rank == 0:
        wandb.finish()
        
if __name__ == "__main__":
    launch_rlg_hydra()
