from typing import Optional
import numpy as np
from abc import abstractmethod

import gymnasium as gym
import ccxt

class UpbitEnvBase(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.action_space = self._get_action_space()
    
    def reset(self, key_path : str = 'up.key', target : str = 'BTC', balance = 100000, seed : Optional[int] = None, options : Optional[dict] = None):
        super().reset(seed=seed)
        
        with open(key_path) as f:
            lines = f.readlines()
            api_key = lines[0].strip()
            api_secret = lines[1].strip()
        
        self.exchange = ccxt.upbit(config={
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        self.target = target
        
        self.ticker = self.exchange.fetch_tickers()[self.target + '/KRW']
    
    def _get_obs(self):
        state = list()
        state.append(self.ticker['high'])
        state.append(self.ticker['low'])
        state.append(self.ticker['vwap'])
        state.append(self.ticker['open'])
        state.append(self.ticker['close'])
        state.append(self.ticker['last'])
        state.append(self.ticker['previousClose'])
        state.append(self.ticker['average'])
        state.append(self.ticker['baseVolume'])
        state.append(self.ticker['quoteVolume'])
        state.append(self.ticker['info']['high_price'])
        state.append(self.ticker['info']['low_price'])
        state.append(self.ticker['info']['signed_change_price'])
        state.append(self.ticker['info']['signed_change_rate'])
        state.append(self.ticker['info']['trade_volume'])
        state.append(self.ticker['info']['acc_trade_price'])
        state.append(self.ticker['info']['acc_trade_price_24h'])
        state.append(self.ticker['info']['acc_trade_volume'])
        state.append(self.ticker['info']['acc_trade_volume_24h'])
        state.append(self.ticker['info']['highest_52_week_price'])
        state.append(self.ticker['info']['lowest_52_week_price'])
        
        # 호가
        self.order_book = self.Exchange.fetch_order_book(symbol=self.TargetCoin+'/KRW')
        for values in self.order_book['asks']:
            for value in values:
                state.append(value)
        
        for values in self.order_book['bids']:
            for value in values:
                state.append(value)
                
        # 현재 지갑
        state.append(self._get_free_krw())
        state.append(self._get_used_krw())
        state.append(self._get_free_coin())
        state.append(self._get_used_coin())
        
        # 오더 정보
        state.append(self._get_buy_order_amount())
        state.append(self._get_buy_order_price())
        state.append(self._get_sell_order_amount())
        state.append(self._get_sell_order_price())
        
        return np.array(state, dtype=np.float32)
    
    def _get_info(self):
        return {
            'TotalAssetValue' : self._calc_total_asset_value()
        }
    
    def _get_current_price_of_coin(self):
        return self.ticker['close']
    
    def _get_free_coin_value(self):
        return self._get_current_price_of_coin() * self._get_free_coin()
    
    def _get_used_coin_value(self):
        return self._get_current_price_of_coin() * self._get_used_coin()
    
    def _calc_total_asset_value(self):
        return self._get_free_krw() + self._get_used_krw() + self._get_free_coin_value() + self._get_used_coin_value()
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def _get_action_space(self):
        pass
    
    @abstractmethod
    def _get_free_krw(self):
        pass
    
    @abstractmethod
    def _get_used_krw(self):
        pass
    
    @abstractmethod
    def _get_free_coin(self):
        pass
    
    @abstractmethod
    def _get_used_coin(self):
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
    
class UpbitSimpleSimulator(UpbitEnvBase):
    class Order():
        def __init__(self, amount, price) -> None:
            self.amount = amount
            self.price = price
            
    def __init__(self, key_path = 'up.key', target = 'BTC'):
        super().__init__(key_path, target)
        
        
    def reset(self, key_path : str = 'up.key', target : str = 'BTC', balance = 100000, seed : Optional[int] = None, options : Optional[dict] = None):
        super().reset(key_path=key_path, target=target, seed=seed, options=options)
        
        self.free_krw = balance
        self.used_krw = 0
        self.free_coin = 0
        self.used_coin = 0
        self.buy_order = None
        self.sell_order = None
        self.prev_asset_value = self._calc_total_asset_value()
        self.terminated_asset_value = balance * 1.5
        self.truncated_asset_value = balance * 0.5
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
        
    
    def step(self, action):
        if action == 1 :    # 구매
            self._create_buy_order(self._get_current_price_of_coin())
        elif action == 2:
            self._create_sell_order(self._get_current_price_of_coin())
        
        curr_asset_value = self._calc_total_asset_value()
        observation = self._get_obs()
        info = self._get_info()
        reward = curr_asset_value - self.prev_asset_value
        self.prev_asset_value = curr_asset_value
        terminated = True if curr_asset_value > self.terminated_asset_value else False
        truncated = True if curr_asset_value < truncated else False
            
        return observation, reward, terminated, truncated, info
        
        
            
        
    
    