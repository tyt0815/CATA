import numpy as np
from collections import deque, namedtuple

import torch

from cata.agents.baseagent import BaseAgent, EDataType
from cata.models.dtqn import DTQN
from cata.agents.buffer import ReplayMemory

Transition = namedtuple('Transition',
                        ('state_obs', 'state_actions', 'action', 'next_state', 'reward'))

class DTQNAgent(BaseAgent):
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
        action_dim = 0,                     # 0으로 설정시 모델이 이전 행동을 고려
        inner_embed_size=128,
        num_heads=8,
        num_layers=2,
        dropout=0.0,
        gate="res",
        identity=False,
        pos="learned",
        bag_size= 0
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
        
        self.memory = ReplayMemory(capacity=self.replay_memory_size, transition=Transition)
        
        discrete=False
        embed_per_obs_dim = 0 # discrete 환경에서만 사용되는 옵션
        vocab_sizes = 0 # discrete 환경에서만 사용되는 옵션
        
        self.policy_net = DTQN(
            obs_dim=self.n_observation,
            num_actions=self.n_action,
            embed_per_obs_dim=embed_per_obs_dim,
            action_dim=action_dim,
            inner_embed_size=inner_embed_size,
            num_heads=num_heads,
            num_layers=num_layers,
            history_len=self.sequence_lenght,
            dropout=dropout,
            gate=gate,
            identity=identity,
            pos=pos,
            discrete=discrete,
            vocab_sizes=vocab_sizes,
            bag_size=bag_size
        )
        self.target_net = DTQN(
            obs_dim=self.n_observation,
            num_actions=self.n_action,
            embed_per_obs_dim=embed_per_obs_dim,
            action_dim=action_dim,
            inner_embed_size=inner_embed_size,
            num_heads=num_heads,
            num_layers=num_layers,
            history_len=self.sequence_lenght,
            dropout=dropout,
            gate=gate,
            identity=identity,
            pos=pos,
            discrete=discrete,
            vocab_sizes=vocab_sizes,
            bag_size=bag_size
        )
        
        print(f"Total Parameters: {self.calc_total_parameters()}")
        
        self.action_memory = deque([0 for _ in range(self.sequence_lenght)], maxlen=self.sequence_lenght)

    def _preprocess_state(self, state):
        state = super()._preprocess_state(state)
        return torch.tensor(np.array(list(state))).unsqueeze(0).to(self.device)
    
    def _forward_network(self, input_state):
        actions = self.policy_net(input_state, torch.tensor(self.action_memory,dtype=torch.float32).view(1,100,1).device(self.device))
        action = torch.argmax(actions[:, -1, :], dim=-1).unsqueeze(0)
        self.action_memory.appendleft(action.item())
        return action
    
    def _set_state_dict_file_path(self):
        self.state_dict_file_path = "./save/dtqn/" + self.state_dict_file_path
        
    def _reset_env(self):
        return self.env.reset(
            sequence_lenght=self.sequence_lenght,
            balance=self.balance,
            end_condition=self.end_condition,
            data_type=self.data_type
        )
        
    # def select_action(self, state):
    #     sample = random.random()
    #     eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
    #     self.steps_done += 1
    #     if sample > eps_threshold:
    #         with torch.no_grad():
    #             return self.policy_net(state).max(1).indices.view(1, 1)
    #     else:
    #         return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        
    # def optimize_model(self):
    #     if len(self.memory) < self.batch_size:
    #         return
    #     transitions = self.memory.sample(batch_size=self.batch_size)
        
    #     batch = Transition(*zip(*transitions))
        
    #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                             batch.next_state)), device=self.device, dtype=torch.bool)
        
    #     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    #     state_batch = torch.cat(batch.state)
    #     action_batch = torch.cat(batch.action)
    #     reward_batch = torch.cat(batch.reward)
        
    #     # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    #     # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    #     state_action_values = self.policy_net(state_batch).gather(1, action_batch)
    #     next_state_values = torch.zeros(self.batch_size, device=self.device)
            
    #     with torch.no_grad():
    #         next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            
    #     # 기대 Q 값 계산
    #     expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    #     # Huber 손실 계산
    #     criterion = nn.SmoothL1Loss()
    #     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    #     # 모델 최적화
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     # 변화도 클리핑 바꿔치기
    #     torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    #     self.optimizer.step()