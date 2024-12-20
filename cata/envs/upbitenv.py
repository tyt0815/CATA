from typing import Optional
import numpy as np
from abc import abstractmethod

import gymnasium as gym
import ccxt

from cata.envs.baseenv import BaseCATAEnv

# TODO: BaseEnv를 상속받는 Env 다시 작성

class BaseUpbitEnv(BaseCATAEnv):
    def __init__(self):
        super().__init__()
        
    def _update_ticker(self):
        try:
            self.ticker = self.exchange.fetch_tickers()[self.target + '/KRW']
        except ccxt.BaseError as e:
            print(f"Fetch tickers error: {str(e)}")
    
    def reset(
            self,
            key_path : str = 'up.key',
            target : str = 'BTC',
            balance = 100000.,
            end_condition : float = 0.1,
            seed : Optional[int] = None,
            options : Optional[dict] = None
        ):
        try:
            with open(key_path) as f:
                lines = f.readlines()
                api_key = lines[0].strip()
                api_secret = lines[1].strip()
        except Exception as e:
            print(f"{key_path} open error: {e}")
            quit()
        
        try:
            self.exchange = ccxt.upbit(config={
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True
            })
        except ccxt.BaseError as e:
            print(f"Log in error: {str(e)}")
            quit()
            
        super().reset(balance=balance, end_condition=end_condition, seed=seed, options=options)
        
        self.target = target        
        self._update_ticker()
        
    
    def step(self):
        self._update_ticker()
        
        return super().step()
    
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
        # self.order_book = self.exchange.fetch_order_book(symbol=self.target +'/KRW')
        # for values in self.order_book['asks']:
        #     for value in values:
        #         state.append(value)
        
        # for values in self.order_book['bids']:
        #     for value in values:
        #         state.append(value)
                
        # 현재 지갑
        state.append(self.free_krw)
        state.append(self.used_krw)
        state.append(self.free_coin)
        state.append(self.used_coin)
        state.append(self._prev_asset_value)
        
        # 오더 정보
        state.append(self._get_buy_order_amount())
        state.append(self._get_buy_order_price())
        state.append(self._get_sell_order_amount())
        state.append(self._get_sell_order_price())
        
        return np.array(state, dtype=np.float32)
      
    
class UpbitSimpleSimulator(BaseUpbitEnv):            
    def __init__(self):
        super().__init__()
        
    def reset(
            self,
            key_path : str = 'up.key',
            target : str = 'BTC',
            balance = 100000.,
            seed : Optional[int] = None,
            options : Optional[dict] = None,
            end_condition : float = 0.1
        ):        
        super().reset(key_path, target, balance, seed, options, end_condition)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
        
    
    def step(self, action):
        '''
        결정론적 환경. 구매시 바로 구매가 되고 매도시 바로 매도가 됨.
        '''
        
        if action == 1 :    # 구매
            self._create_buy_order(self._get_current_price_of_coin())
        elif action == 2:   # 매도
            self._create_sell_order(self._get_current_price_of_coin())
        
        return super().step()
    
    def _get_action_space(self):
        return gym.spaces.Discrete(3)
    
    def _update_free_krw(self):
        pass
    
    def _update_used_krw(self):
        pass
    
    def _update_free_coin(self):
        pass
    
    def _update_used_coin(self):
        pass
    
    def _get_buy_order_amount(self):
        return 0
    
    def _get_buy_order_price(self):
        return 0
    
    def _get_sell_order_amount(self):
        return 0
    
    def _get_sell_order_price(self):
        return 0
    
    def _create_buy_order(self, price):
        transaction_amount = self.free_krw / price
        self.free_coin += transaction_amount
        self.free_krw -= transaction_amount * price
    
    def _create_sell_order(self, price):
        transaction_amount = self.free_coin * price
        self.free_krw += transaction_amount
        self.free_coin -= transaction_amount / price
    
    def _cancle_order(self):
        pass