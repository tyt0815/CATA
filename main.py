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

from cata.envs.upbitenv import UpbitSimpleSimulator
from cata.envs.offlineenv import OfflineCATAEnv, EDataType

from cata.agents.dqnagent import DQNAgent
from cata.agents.drqnagent import DRQNAgent
from cata.agents.dtqnagent import DTQNAgent

AGENT_MAP = {
    'DQN'   :   DQNAgent,
    'DRQN'  :   DRQNAgent,
    'DTQN'  :   DTQNAgent
}

ENV_MAP = ["offline"]

TARGET_MAP = ["BTC"]

DATA_TYPE_MAP = {
    "CloseOnly" : EDataType.close_only.value,
    "All" : EDataType.all.value
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--agent',
        type=str,
        default='DRQN',
        choices=list(AGENT_MAP.keys())
    )
    parser.add_argument(
        "--env",
        type=str,
        default='offline',
        choices=ENV_MAP
    )
    
    parser.add_argument(
        "--target",
        type=str,
        default='BTC',
        choices=TARGET_MAP
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=128
    )
    
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99
    )
    
    parser.add_argument(
        "--epsstart",
        type=float,
        default=0.9
    )
 
    parser.add_argument(
        "--epsend",
        type=float,
        default=0.05
    )
    
    parser.add_argument(
        "--epsdecay",
        type=int,
        default=1000
    )
    
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )
    
    parser.add_argument(
        "--sequence",
        type=int,
        default=100
    )
    
    parser.add_argument(
        "--balance",
        type=int,
        default=100000
    )
    
    parser.add_argument(
        "--endcondition",
        type=float,
        default=0.1
    )
    
    parser.add_argument(
        "--datatype",
        type=str,
        default="CloseOnly",
        choices=list(DATA_TYPE_MAP.keys())
    )
    
    parser.add_argument(
        "--replaymemory",
        type=int,
        default=10000
    )
    
    parser.add_argument(
        "--episode",
        type=int,
        default=100
    )
    
    parser.add_argument(
        "--iter",
        type=int,
        default=0
    )
    
    parser.add_argument(
        "--normalize",
        type=bool,
        default=True
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)  
    
    if args.env == "offline":
        env = OfflineCATAEnv("BTC_Data.csv")
        
    target_coin = args.target
    BATCH_SIZE = args.batch
    GAMMA = args.gamma
    EPS_START = args.epsstart
    EPS_END = args.epsend
    EPS_DECAY = args.epsdecay
    TAU = args.tau
    LR = args.lr
    
    sequence_lenght = args.sequence
    balance = args.balance
    end_condition = args.endcondition
    data_type = DATA_TYPE_MAP[args.datatype]
    replay_memory_size = args.replaymemory
    
    n_episodes = args.episode
    n_iter = args.iter
    
    if args.agent == "DRQN":
        agent = DRQNAgent(
            env=env,
            device=device,
            eps_start = EPS_START,
            eps_end = EPS_END,
            eps_decay = EPS_DECAY,
            batch_size = BATCH_SIZE,
            gamma = GAMMA,
            tau = TAU,
            lr = LR,
            sequence_lenght = sequence_lenght,
            balance = balance,
            end_condition= end_condition,
            data_type = data_type,
            replay_memory_size = replay_memory_size,
            hidden_size=128,
            num_layers=64,
            bnormalized=args.normalize
        )
        
    agent.train(n_episodes=n_episodes, n_iter=n_iter)