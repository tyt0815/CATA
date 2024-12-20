import numpy as np

import torch
import torch.optim as optim

from cata.agents.baseagent import BaseAgent, EDataType
from cata.models.lstm import LSTM


class DRQNAgent(BaseAgent):
    
    def __init__(
        self,
        env,
        device,
        bnormalized,
        eps_start = 0.9,
        eps_end = 0.05,
        eps_decay = 1000,
        batch_size = 128,
        gamma = 0.99,
        tau = 0.005,
        lr = 1e-4,
        sequence_lenght = 100,
        balance = 100000,
        end_condition : float = 0.1,
        data_type = EDataType.all.value,
        replay_memory_size = 10000,
        hidden_size=128,
        num_layers=64
    ):
        
        super().__init__(
            env=env,
            device=device,
            bnormalized=bnormalized,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            lr=lr,
            sequence_lenght=sequence_lenght,
            balance=balance,
            end_condition=end_condition,
            data_type=data_type,
            replay_memory_size=replay_memory_size
        )
        
        self.policy_net = LSTM(input_size=self.n_observation, hidden_size=hidden_size, num_layers=num_layers, output_size=self.n_action).to(device=self.device)
        self.target_net = LSTM(input_size=self.n_observation, hidden_size=hidden_size, num_layers=num_layers, output_size=self.n_action).to(device=self.device)
        print(f"Total Parameters: {self.calc_total_parameters()}")

    def _preprocess_state(self, state):
        state = super()._preprocess_state(state)
        return torch.tensor(np.array(list(state[0]))).unsqueeze(0).to(self.device)
    
    def _set_state_dict_file_path(self):
        self.state_dict_file_path = "./save/drqn/" + self.state_dict_file_path