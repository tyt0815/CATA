from abc import abstractmethod
from collections import namedtuple
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from itertools import count
import time
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim

from cata.agents.buffer import ReplayMemory
from cata.envs.baseenv import EDataType

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class BaseAgent:
    
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
        replay_memory_size = 10000
        ):
        print("Initialize Agent...", end=' ')
        self.env = env
        self.device = device
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.bnormalized = bnormalized
        
        self.sequence_lenght = sequence_lenght
        self.balance = balance
        self.end_condition = end_condition
        self.data_type = data_type
        states, infos = self._reset_env()
        self.n_action = self.env.action_space.n
        self.n_observation = len(states[0][0])
        
        self.steps_done = 0
        self.episode_durations = []
        self.replay_memory_size = replay_memory_size
        self.memory = ReplayMemory(capacity=self.replay_memory_size, transition=Transition)
        
        self.state_dict_file_path = f"batch{self.batch_size}_sequence{self.sequence_lenght}_datatype{self.data_type}.pth"
        self._set_state_dict_file_path()
        
    def _reset_env(self):
        return self.env.reset(
            sequence_lenght=self.sequence_lenght,
            balance=self.balance,
            end_condition=self.end_condition,
            data_type=self.data_type
        )
        
    def calc_total_parameters(self):
        return sum(p.numel() for p in self.policy_net.parameters())
     
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        
    def plot_durations(self, show_result = False):
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.clf()
        # plt.ylim(-5, 5)
        if show_result:
            plt.title('Result')
        else:
            plt.title('Training')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.plot(durations_t.numpy())
        
        plt.draw()
        plt.pause(0.000001)
        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(batch_size=self.batch_size)
        
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
        # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
            
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Huber 손실 계산
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        # 변화도 클리핑 바꿔치기
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def train(self, n_episodes=0, n_iter=0):
        self.steps_done = 0
        plt.ion()
            
        if os.path.exists(self.state_dict_file_path):
            print("load state dict")
            self.policy_net.load_state_dict(torch.load(self.state_dict_file_path))
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
            
        self.total_i = 0
        self.episode_durations = []
        
        if n_episodes == 0: 
            episode_iter = count()
        else:
            episode_iter = range(n_episodes)
        for i_episode in episode_iter:
            state, info = self._reset_env()
            state = self._preprocess_state(state=state)
            cumulative_reward = 0
            self.episode_durations.append(0)
            
            if n_iter == 0:
                inner_iter = count()
            else:
                inner_iter = range(n_iter)
                
            for i_iter in inner_iter:
                
                start = time.time()
                action = self.select_action(state=state)
                observation, reward, terminated, truncated, info = self.env.step(action.item())
                cumulative_reward += reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = self._preprocess_state(state=observation)
                    
                self.memory.push(state, action, next_state, reward)
                
                state = next_state
                self.optimize_model()
                
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)
                
                end = time.time()
                
                self.episode_durations[-1] = cumulative_reward
                self.plot_durations()
                
                self.total_i += 1
                sys.stdout.write(f"\r{f'episode {i_episode + 1} - {i_iter + 1}':<17}  reward: {cumulative_reward:<20}  elapsed time: {end - start:<20}      \n")
                
                if done:
                    break
                sys.stdout.write("\033[F\033[K")
                
            torch.save(self.policy_net.state_dict(), self.state_dict_file_path)
        
        self.plot_durations(show_result=True)
        plt.ioff()
        plt.show()
            
    def _preprocess_state(self, state):
        # 정규화
        if self.bnormalized:
            if self.data_type == EDataType.all.value:
                market_data_len = 5
            elif self.data_type == EDataType.close_only.value:
                market_data_len = 1
            
            else:
                raise Exception("error: undefined")
            
            market_state = state[0][:,:market_data_len]
            balance_state = state[0][:,market_data_len:]
            
            min_vals = np.min(market_state, axis=0)
            max_vals = np.max(market_state, axis=0)
            
            denominator = max_vals - min_vals

            # 값이 모두 같은 경우, 정규화 결과를 0으로
            # denominator[denominator == 0] = 1  # 0인 경우 1로 대체 (정규화 결과 0)
            # market_state = (market_state - min_vals) / denominator
            
            # 값이 모두 같은 경우, 정규화 결과를 0.5로
            market_state = np.where(
                denominator == 0, 
                0.5,  # 동일 값인 경우 0.5로 설정
                (market_state - min_vals) / denominator
            )
            
            
            # 설계상, 모든 값이 같을 수 없음.
            min_val = np.min(balance_state)
            max_val = np.max(balance_state)

            # 정규화 공식 적용
            balance_state = (balance_state - min_val) / (max_val - min_val)
            
            state[0] = np.hstack((market_state, balance_state))
        
        return state
    
    @abstractmethod
    def _set_state_dict_file_path(self):
        pass
