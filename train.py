import sys
import random
import math
import argparse
import matplotlib
import matplotlib.pyplot as plt
import time
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from cata.envs.UpbitEnvironment import UpbitSimpleSimulator
from cata.models.mlp import MLP
from cata.models.lstm import LSTM

MODEL_MAP = {
    'DQN' : MLP,
    'DRQN' : LSTM
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='DRQN',
        choices=list(MODEL_MAP.keys())
    )
    
    return parser.parse_args()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
            return len(self.memory)
    
steps_done = 0    
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations = []
def plot_durations(show_result = False):
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.plot(durations_t.numpy())
    
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())
    plt.draw()
    plt.pause(0.5)
    plt.clf()
        
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(batch_size=BATCH_SIZE)
    
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    # 변화도 클리핑 바꿔치기
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if __name__ == '__main__':
    args = get_args()
    
    print("Initialize...")
    plt.ion()
    
    env = UpbitSimpleSimulator()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)    
    
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    target_coin = 'BTC'

    n_action = env.action_space.n
    state, info = env.reset()
    n_input_state = 2
    n_observation = len(state)

    if args.model == 'DQN':
        policy_net = MLP(n_observations=n_observation * n_input_state, n_actions=n_action).to(device=device)
        target_net = MLP(n_observations=n_observation* n_input_state, n_actions=n_action).to(device=device)
    elif args.model == 'DRQN':
        policy_net = LSTM(input_size=n_observation, hidden_size=128, num_layers=64, output_size=n_action).to(device=device)
        target_net = LSTM(input_size=n_observation, hidden_size=128, num_layers=64, output_size=n_action).to(device=device)    
    
    total_params = sum(p.numel() for p in policy_net.parameters())
    print(f"Total number of parameters: {total_params}")
    
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    if torch.cuda.is_available():
        num_episodes = 600
        n_iter = BATCH_SIZE * 2
    else:
        num_episodes = 50
        n_iter = 50
        
    state_deque = deque(maxlen=n_input_state)
       
        
    for i_episode in range(num_episodes):
        # 환경과 상태 초기화
        state, _ = env.reset(target=target_coin, end_condition=0.01)
        for i in range(n_input_state):
            next_state, _, _, _, _ = env.step(0)
            state_deque.append(next_state) 
        if args.model == 'DQN':
            state = torch.cat([torch.tensor(s) for s in list(state_deque)]).unsqueeze(0).to(device)
        elif args.model == 'DRQN':
            state = torch.tensor(np.array(list(state_deque))).unsqueeze(0).to(device)
        
        cumulative_reward = 0
        for t in range(n_iter):
            start = time.time()
            action = select_action(state)
            end = time.time()
            observation, reward, terminated, truncated, info = env.step(action.item())
            cumulative_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                state_deque.append(observation)
                if args.model == 'DQN':
                    next_state = torch.cat([torch.tensor(s) for s in list(state_deque)]).unsqueeze(0).to(device)
                elif args.model == 'DRQN':
                    next_state = torch.tensor(np.array(list(state_deque))).unsqueeze(0).to(device=device)
                # next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # 메모리에 변이 저장
            memory.push(state, action, next_state, reward)

            # 다음 상태로 이동
            state = next_state

            # (정책 네트워크에서) 최적화 한단계 수행
            optimize_model()

            # 목표 네트워크의 가중치를 소프트 업데이트
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            sys.stdout.write(f"\rEpisode{i_episode + 1:>3} - {t + 1:<3}\n")
            sys.stdout.write(f"{'Response Time':<20} {end - start:<25}\n")
            sys.stdout.write(f"{'Action:':<20} {action[0][0]:<25}\n")
            sys.stdout.write(f"{'Reward:':<20} {reward[0]:<25}\n")
            sys.stdout.write(f"{'Cumulative Reward:':<20} {cumulative_reward:<25}\n")
            sys.stdout.write(f"{f'Current {target_coin} Price:':<20} {info['current_price']:<25}\n")
            sys.stdout.write(f"{'Prev Asset Value:':<20} {info['prev_asset_value']:<25}\n")
            sys.stdout.write(f"{'Current Asset Value:':<20} {info['curr_asset_value']:<25}\n")
            sys.stdout.write(f"{'Free KRW:':<20} {info['free_krw']:<25}\n")
            sys.stdout.write(f"{'Used KRW:':<20} {info['used_krw']:<25}\n")
            sys.stdout.write(f"{'Free Coin:':<20} {info['free_coin']:<25}\n")
            sys.stdout.write(f"{'Used Coin:':<20} {info['used_coin']:<25}\n")
            sys.stdout.write(f"{'Buy Order Amount:':<20} {info['buy_order_amount']:<25}\n")
            sys.stdout.write(f"{'Buy Order Price:':<20} {info['buy_order_price']:<25}\n")
            sys.stdout.write(f"{'Sell Order Amount:':<20} {info['sell_order_amount']:<25}\n")
            sys.stdout.write(f"{'Sell Order Price:':<20} {info['sell_order_price']:<25}\n")
            sys.stdout.write(f"DEBUG: {next_state.shape}\n")
                
            
            for i in range(17): sys.stdout.write("\033[F")
            sys.stdout.flush()

            # if done:
        episode_durations.append(cumulative_reward)
        plot_durations()
                # break
        

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()