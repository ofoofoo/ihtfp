from constants import EURIKA_ROOT_DIR

class PromptData:
    def __init__(self, cfg):
        self.task_file = f'{EURIKA_ROOT_DIR}/envs/{cfg.training_filename}'
        self.task_code_string  = self.file_to_string(self.task_file)

        prompt_dir = f"{EURIKA_ROOT_DIR}/utils/prompts"
        self.initial_system = self.file_to_string(f'{prompt_dir}/initial_system.txt')
        self.code_output_tip = self.file_to_string(f'{prompt_dir}/code_output_tip.txt')
        self.code_feedback = self.file_to_string(f'{prompt_dir}/code_feedback.txt')
        self.initial_user = self.file_to_string(f'{prompt_dir}/initial_user.txt')
        self.reward_signature = self.file_to_string(f'{prompt_dir}/reward_signature_{cfg.task.lower()}.txt')
        self.policy_feedback = self.file_to_string(f'{prompt_dir}/policy_feedback.txt')
        self.execution_error_feedback = self.file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

        
        self.initial_system = self.initial_system.format(task_reward_signature_string=self.reward_signature) + self.code_output_tip
        self.initial_user = self.initial_user.format(task_obs_code_string=self.task_code_string, task_description=cfg.task_description)

        self.code_feedback = self.file_to_string(f'{prompt_dir}/code_feedback.txt')


        self.messages = [{"role": "system", "content": self.initial_system}, {"role": "user", "content": self.initial_user}]

    def file_to_string(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()
