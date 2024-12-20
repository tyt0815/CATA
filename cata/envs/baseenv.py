import gymnasium as gym
from typing import Optional
from abc import abstractmethod
from collections import deque
import numpy as np

from enum import Enum, auto
# TODO: BaseEnv 클래스 작성

class EAction(Enum):
    idle = 0
    buy_order = auto()
    sell_order = auto()  

class EDataType(Enum):
    all = 0
    close_only = auto()
      
class BaseCATAEnv(gym.Env):
    
    def __init__(self):
        super().__init__()
        self.action_space = self._get_action_space()
        
    def reset(
            self,
            sequence_lenght = 100,
            balance = 100000,
            end_condition : float = 0.1,
            data_type = EDataType.all.value,
            seed : Optional[int] = None,
            options : Optional[dict] = None,
        ):
        super().reset(seed=seed)
        
        self.sequence_lenght = sequence_lenght
        self.free_krw = balance
        self.used_krw = 0
        self.free_coin = 0
        self.used_coin = 0
        self.curr_asset_value = balance
        # self._prev_asset_value = balance
        self._buy_value = 0
        self._sell_value = 0
        self.terminated_asset_value = balance * (1 + end_condition)
        self.truncated_asset_value = balance * (1 - end_condition)
        self.data_type = data_type
        
        self.action_history = deque([[0] for _ in range(self.sequence_lenght)], maxlen=self.sequence_lenght)
    
    def step(self, action):
        # self._prev_asset_value = self.curr_asset_value
        self.curr_asset_value = self._calc_total_asset_value()
        
        self.action_history.appendleft([action])
        
        return self._get_obs(), self._calc_reward(), self._is_terminated(), self._is_truncated(), self._get_info()
    
    def _get_info(self):
        return {
            'current_price' : self._get_current_price_of_coin(),
            # 'prev_asset_value' : self._prev_asset_value,
            'curr_asset_value' : self._calc_total_asset_value(),
            'free_krw' : self.free_krw,
            'used_krw' : self.used_krw,
            'free_coin' : self.free_coin,
            'used_coin' : self.used_coin,
            'buy_order_amount': self._get_buy_order_amount(),
            'buy_order_price': self._get_buy_order_price(),
            'sell_order_amount' : self._get_sell_order_amount(),
            'sell_order_price': self._get_sell_order_price()
        }
        
    def _exchange_from_coin_to_krw(self, coin):
        return self._get_current_price_of_coin() * coin
    
    def _exchange_from_krw_to_coin(self, krw):
        return self._get_current_price_of_coin() / krw
    
    def _get_free_coin_value(self):
        return self._exchange_from_coin_to_krw(self.free_coin)
    
    def _get_used_coin_value(self):
        return self._exchange_from_coin_to_krw(self.used_coin)
    
    def _calc_total_asset_value(self):
        return self.free_krw + self.used_krw + self._get_free_coin_value() + self._get_used_coin_value()
        
    def _calc_reward(self):
        reward = 0
        if self._sell_value > 0:
            reward = (self._sell_value - self._buy_value) / self._buy_value * 100
            self._sell_value = 0
            self._buy_value = 0
        return reward
        
    def _is_terminated(self):
        return True if self.curr_asset_value > self.terminated_asset_value else False
    
    def _is_truncated(self):
        return True if self.curr_asset_value < self.truncated_asset_value else False
    
    def _get_obs(self):
        if self.data_type == EDataType.all.value:
            state = self._get_obs_all()
            
        elif self.data_type == EDataType.close_only.value:
            state = self._get_obs_close_only()
        
        else:
            raise Exception("BaseCATAEnv._get_obs error")
        
        self._append_balance_obs(state=state)
        self.sequence.appendleft(state)
        return [np.array(self.sequence, dtype=np.float32), np.array(self.action_history, dtype=np.float32)]
        
    def _append_balance_obs(self, state):
        state.append(self.free_krw)
        state.append(self.used_krw)
        state.append(self._exchange_from_coin_to_krw(self.free_coin))
        state.append(self._exchange_from_coin_to_krw(self.used_coin))
        state.append(self._buy_value)
    
    @abstractmethod
    def _get_obs_all(self):
        pass
    
    @abstractmethod
    def _get_obs_close_only(self):
        pass
        
    @abstractmethod
    def _get_action_space(self):
        pass
    
    @abstractmethod
    def _get_current_price_of_coin(self):
        pass
    
    @abstractmethod
    def _get_buy_order_amount(self):
        pass
    
    @abstractmethod
    def _get_buy_order_price(self):
        pass
    
    @abstractmethod
    def _get_sell_order_amount(self):
        pass
    
    @abstractmethod
    def _get_sell_order_price(self):
        pass
    
    @abstractmethod
    def _create_buy_order(self, price):
        pass
    
    @abstractmethod
    def _create_sell_order(self, price):
        pass
    
    @abstractmethod
    def _cancle_order(self):
        pass
    