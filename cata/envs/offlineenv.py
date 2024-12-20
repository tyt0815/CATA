from typing import Optional
import pandas as pd
import numpy as np
from collections import deque

import gymnasium as gym

from cata.envs.baseenv import BaseCATAEnv, EAction, EDataType

class OfflineCATAEnv(BaseCATAEnv):
    
    def __init__(self, file_path : str):
        super().__init__()
        self.file_path = file_path
    
    def reset(
        self,
        sequence_lenght = 100,
        balance = 100000,
        end_condition : float = 0.1,
        data_type = EDataType.all.value,
        use_action_history = False,
        seed : Optional[int] = None,
        options : Optional[dict] = None,
    ):
        super().reset(
            sequence_lenght = sequence_lenght,
            balance = balance,
            end_condition = end_condition,
            data_type=data_type,
            seed = seed,
            options = options
            )
        
        self.idx = -1
        try:
            self.data = pd.read_csv(self.file_path)
        except Exception as e:
            print(f"Data file read error: {e}")
            quit()
            
        self.sequence = deque([], maxlen=self.sequence_lenght)
        
        for _ in range(self.sequence_lenght - 1):
            self._get_obs()
            
        return self._get_obs(), self._get_info()
            
    def step(self, action):
        # 결정론적 환경. 구매 혹은 판매시 바로바로 시행
        if action == EAction.buy_order.value:
            self._create_buy_order(self._get_current_price_of_coin())
        elif action == EAction.sell_order.value:
            self._create_sell_order(self._get_current_price_of_coin())
            
        return super().step(action)
    
    def _is_truncated(self):
        return super()._is_truncated() or self.idx + 1 == len(self.data)
    
    def _get_obs(self):
        self.idx += 1
        return super()._get_obs()
    
    def _get_obs_all(self):
        state = list()
        state.append(self.data['open'][self.idx])
        state.append(self.data['high'][self.idx])
        state.append(self.data['low'][self.idx])
        state.append(self.data['close'][self.idx])
        state.append(self.data['baseVolume'][self.idx])
        return state
    
    def _get_obs_close_only(self):
        state = list()
        state.append(self.data['close'][self.idx])
        return state
        
    def _get_action_space(self):
        return gym.spaces.Discrete(3)
    
    def _get_current_price_of_coin(self):
        return self.data["close"][self.idx]
    
    def _get_buy_order_amount(self):
        return 0
    
    def _get_buy_order_price(self):
        return 0
    
    def _get_sell_order_amount(self):
        return 0
    
    def _get_sell_order_price(self):
        return 0
    
    def _create_buy_order(self, price):
        self._buy_value += self.free_krw
        self.free_coin += (self.free_krw / price)
        self.free_krw = 0
    
    def _create_sell_order(self, price):
        self._sell_value += (price * self.free_coin)
        self.free_krw += (price * self.free_coin)
        self.free_coin = 0
    
    def _cancle_order(self):
        pass
        