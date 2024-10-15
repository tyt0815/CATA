from typing import Optional
import numpy as np
from abc import abstractmethod

import gymnasium as gym
import ccxt

class UpbitEnvBase(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = self._get_action_space()
        
    def _update_ticker(self):
        self.ticker = self.exchange.fetch_tickers()[self.target + '/KRW']
    
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
        self._update_ticker()
        self.prev_asset_value = self._calc_total_asset_value()
        self.terminated_asset_value = balance * 1.5
        self.truncated_asset_value = balance * 0.5
        
    def _end_reset(self):
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self):
        self._update_ticker()
    
    def _end_step(self):
        curr_asset_value = self._calc_total_asset_value()
        observation = self._get_obs()
        info = self._get_info()
        reward = curr_asset_value - self.prev_asset_value
        self.prev_asset_value = curr_asset_value
        terminated = True if curr_asset_value > self.terminated_asset_value else False
        truncated = True if curr_asset_value < self.truncated_asset_value else False
            
        return observation, reward, terminated, truncated, info
    
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
        self.order_book = self.exchange.fetch_order_book(symbol=self.target +'/KRW')
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
            'current_price' : self._get_current_price_of_coin(),
            'total_asset_value' : self._calc_total_asset_value(),
            'free_krw' : self._get_free_krw(),
            'used_krw' : self._get_used_krw(),
            'free_coin' : self._get_free_coin(),
            'used_coin' : self._get_used_coin(),
            'buy_order_amount': self._get_buy_order_amount(),
            'buy_order_price': self._get_buy_order_price(),
            'sell_order_amount' : self._get_sell_order_amount(),
            'sell_order_price': self._get_sell_order_price()
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
    
    @abstractmethod
    def _cancle_order(self):
        pass
    
class UpbitSimpleSimulator(UpbitEnvBase):            
    def __init__(self):
        super().__init__()
        
    def reset(self, key_path : str = 'up.key', target : str = 'BTC', balance = 100000, seed : Optional[int] = None, options : Optional[dict] = None):
        self.free_krw = balance
        self.used_krw = 0
        self.free_coin = 0
        self.used_coin = 0
        
        super().reset(key_path=key_path, target=target, seed=seed, options=options)
        
        return self._end_reset()
        
    
    def step(self, action):
        '''
        결정론적 환경. 구매시 바로 구매가 되고 매도시 바로 매도가 됨.
        '''
        super().step()
        
        if action == 1 :    # 구매
            self._create_buy_order(self._get_current_price_of_coin())
        elif action == 2:   # 매도
            self._create_sell_order(self._get_current_price_of_coin())
                    
        return self._end_step()
    
    def _get_action_space(self):
        return 3
    
    def _get_free_krw(self):
        return self.free_krw
    
    def _get_used_krw(self):
        return self.used_krw
    
    def _get_free_coin(self):
        return self.free_coin
    
    def _get_used_coin(self):
        return self.used_coin
    
    def _get_buy_order_amount(self):
        return 0
    
    def _get_buy_order_price(self):
        return 0
    
    def _get_sell_order_amount(self):
        return 0
    
    def _get_sell_order_price(self):
        return 0
    
    def _create_buy_order(self, price):
        self.free_coin = self.free_krw / price
        self.free_krw = 0
    
    def _create_sell_order(self, price):
        self.free_krw = self.free_coin * price
        self.free_coin = 0
    
    def _cancle_order(self):
        pass