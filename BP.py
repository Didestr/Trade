import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
#import telebot
import io
import math
import time
import datetime
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO
from binance.um_futures import UMFutures
from datetime import timedelta, timezone
from PIL import Image
from tinkoff.invest import CandleInterval, Client, PortfolioResponse, PortfolioPosition, AccessLevel, RequestError, OrderDirection, OrderType, Quotation
from tinkoff.invest.utils import now, quotation_to_decimal
from tinkoff.invest.services import InstrumentsService, MarketDataService, Services

TOKEN = os.environ["TINKOFF_API"]
token = os.environ["TELEGRAM_TOKEN"]
chat_id = os.environ["CHAT_ID"]
admin_chat_id = os.environ["ADMIN_CHAT_ID"]
account_id = os.environ["ACCOUNT_ID"]


class Candle:
    def __init__(self, symbol:str, interval='1d', limit=365) -> None:
        assert interval in ['1m', '5m', '15m', '1h', '1d']
        self.df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        if len(symbol) == 12 or len(symbol) == 4:
            if interval == '1m':
                interval = 1
                mult = 1 / 60
            elif interval == '5m':
                interval = 2
                mult = 1 / 60 * 5
            elif interval == '15m':
                interval = 3
                mult = 1 / 60 * 15
            elif interval == '1h':
                interval = 4
                mult = 1
            elif interval == '1d':
                interval = 5
                mult = 24
            if len(symbol) == 4:
                symbol = self.get_figi(symbol)
            self.from_tinkoff(symbol, interval, limit * mult)
        elif 4 < len(symbol) < 12:
            self.from_binance(symbol, interval, limit)

    def from_binance(self, symbol= 'BTCUSDT', interval='1d', limit=365) -> pd.DataFrame:
        um_futures_client = UMFutures()
        res = um_futures_client.klines(symbol, interval, limit = limit)
        data = pd.DataFrame(res, columns=['time', 'open', 'high', 'low', 'close', 'volume', '1', '2', '3', '4', '5', '6'])
        data = data.drop(['1', '2', '3', '4', '5', '6'], axis=1)
        self.df['Date'] = pd.to_datetime(data['time'], unit='ms', utc=True)
        self.df['Open'] = pd.to_numeric(data['open'], errors='coerce')
        self.df['High'] = pd.to_numeric(data['high'], errors='coerce')
        self.df['Low'] = pd.to_numeric(data['low'], errors='coerce')
        self.df['Close'] = pd.to_numeric(data['close'], errors='coerce')
        self.df['Volume'] = pd.to_numeric(data['volume'], errors='coerce')
        self.df.set_index('Date', inplace=True)
        return self.df

    def get_figi(self, symbol):
        with Client(TOKEN) as cl:
            instruments: InstrumentsService = cl.instruments
    
            l = []
            for method in ['shares', 'currencies', 'bonds', 'etfs', 'futures']:
                for item in getattr(instruments, method)().instruments:
                    l.append({
                        'ticker': item.ticker,
                        'figi': item.figi
                    })
    
            df = pd.DataFrame(l)
            df = df[df['ticker'] == symbol]
            df = df.reset_index(drop=True)
            return df['figi'].loc[0]

    def from_tinkoff(self, figi = 'BBG004730N88', interval = 5, hours = 365) -> pd.DataFrame:
        with Client(TOKEN) as client:
            for candle in client.get_all_candles(
                figi = figi,
                from_ = now() - timedelta(hours=hours),
                interval=interval,
            ):
                data1 = pd.DataFrame({'Date': [candle.time],
                                    'Open': [float(quotation_to_decimal(candle.open))],
                                    'High': [float(quotation_to_decimal(candle.high))],
                                    'Low': [float(quotation_to_decimal(candle.low))],
                                    'Close': [float(quotation_to_decimal(candle.close))],
                                    'Volume': [float(candle.volume)]})
                self.df = pd.concat([self.df, data1], ignore_index=True)
        self.df.set_index('Date', inplace=True)
        return self.df
    
    def show(self):
        print(self.df)

    def draw(self):
        plt.figure(1, figsize=(16,6))
        up = self.df[self.df.Close >= self.df.Open]
        down = self.df[self.df.Close < self.df.Open]
        col1 = 'red'
        col2 = 'green'
        width = .6
        width2 = .06
        plt.bar(up.index, up.Close-up.Open, width, bottom=up.Open, color=col1)
        plt.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color=col1)
        plt.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color=col1)
        plt.bar(down.index, down.Close-down.Open, width, bottom=down.Open, color=col2)
        plt.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color=col2)
        plt.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color=col2)
        plt.xticks(rotation=30, ha='right')
        if 'EMA' in self.df:
            plt.plot(self.df['EMA'])
        if 'SMA' in self.df:
            plt.plot(self.df['SMA'])
        if 'RSI' in self.df:
            plt.figure(2, figsize=(16,3))
            plt.plot(self.df['RSI'])
        if 'OBV' in self.df:
            plt.figure(3, figsize=(16,3))
            plt.plot(self.df['OBV'])
        plt.show()
        
    def add_EMA(self, len):
        self.df['EMA'] = ta.ema(self.df['Close'], len)

    def add_SMA(self, len):
        self.df['SMA'] = ta.sma(self.df['Close'], len)

    def add_RSI(self, len):
        self.df['RSI'] = ta.rsi(self.df['Close'], len)
                                
    def add_OBV(self):
        self.df['OBV'] = ta.obv(self.df['Close'], self.df['Volume'])

class TradingEnvironment(gym.Env):
    def __init__(self, df):
        super(TradingEnvironment, self).__init__()

        # Define the observation space
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(9,), dtype=np.float64)

        # Define the action space
        self.action_space = gym.spaces.Discrete(4)  # 0: Buy, 1: Sell, 2: Neutral, 3: Close buy, 4: Close sell

        # Load your dataframe here
        self.df = df

        # Set the initial state
        self.current_step = 14

        self.order_vol = 100
        self.order_limit = 5
        self.profit = 0
        self.long_stack = []
        self.short_stack = []

    def reset(self):
        # Reset the current step and return the initial observation
        self.current_step = 14
        self.profit = 0
        self.long_stack = []
        self.short_stack = []
        
        return self._get_observation()

    def step(self, action):
        # Execute the given action
        reward = self._execute_action(action)
        done = False
        # Update the current step
        self.current_step += 1

        # Check if the episode is done
        if self.current_step >= len(self.df):
            done = True
            self.current_step = 14

        # Return the next observation, the reward, and whether the episode is done
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # Return the current observation based on the current step
        return self.df.iloc[self.current_step].values

    def _execute_action(self, action):
        # Implement the logic for executing the given action
        # You will need to define how the action affects the environment and calculate the reward
        # You can access the current state using self.df.iloc[self.current_step]
        # Based on the action, update the environment and calculate the reward
        # Return the reward

        # Example logic:
        all_price = 0
        quantity = 0
        count = 0 
        sum_price = 0
        current_price = self.df["Close"].iloc[self.current_step]
        for i in range(1,6):
            sum_price += self.df["Close"].iloc[self.current_step - i]

        prew_cur_price = sum_price / 5
        reward = 0

        if action == 2:  # Close buy
            if self.long_stack:
                for price in self.long_stack:
                    count += 1
                    all_price += price
                    quantity += self.order_vol / price
                avg_price_entry = all_price / count
                reward = (current_price - avg_price_entry) * quantity
                self.long_stack.clear()

        elif action == 3:  # Close sell
            if self.short_stack:
                for price in self.short_stack:
                    count += 1
                    all_price += price
                    quantity += self.order_vol / price
                avg_price_entry = all_price / count
                reward = (avg_price_entry - current_price) * quantity
                self.short_stack.clear()

        elif action == 0:  # Buy
            if len(self.long_stack) < self.order_limit:
                reward = prew_cur_price - current_price
                self.long_stack.append(current_price)

        elif action == 1:  # Sell
            if len(self.short_stack) < self.order_limit:
                reward = current_price - prew_cur_price
                self.short_stack.append(current_price)

        return reward

def main(): 
    model_dit = "models/A2C"
    logdir = "logs"

    if not os.path.exists(model_dit):
        os.makedirs(model_dit)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    BTCUSDT = Candle('SBER', '1h', limit=20000)
    BTCUSDT.add_EMA(14)
    BTCUSDT.add_RSI(14)
    BTCUSDT.add_SMA(14)
    BTCUSDT.add_OBV()
    #BTCUSDT.draw()
    
    df = BTCUSDT.df
    df = df.reset_index(drop=True)
    df=(df-df.mean())/df.std()
    
    print(df)


    env = TradingEnvironment(df)

    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    TIMESTEP = 10000
    for i in range(1, 10):
        model.learn(total_timesteps=TIMESTEP, reset_num_timesteps=False, tb_log_name="A2C")
        model.save(f"{model_dit}/{TIMESTEP*i}")
    # Use the trained model for prediction
    obs = env.reset()
    actions = []
    rewards_list = []
    rewards = 0

    for _ in range(len(df)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        actions.append(action)
        rewards_list.append(reward)
        rewards += reward

        if done:
            obs = env.reset()

    print(rewards_list)
    print(rewards)
    
    # Plotting the closed price chart
    plt.plot(df["Close"], label="Closed Price")

    # Plotting the buy and sell points
    buys = [i for i in range(len(df)) if actions[i] == 0]
    sells = [i for i in range(len(df)) if actions[i] == 1]
    close_buys = [i for i in range(len(df)) if actions[i] == 2]
    close_sells = [i for i in range(len(df)) if actions[i] == 3]
    plt.scatter(buys, df["Close"].iloc[buys], color='green', label='Buy')
    plt.scatter(sells, df["Close"].iloc[sells], color='red', label='Sell')
    plt.scatter(close_buys, df["Close"].iloc[close_buys], color='aqua', label='close Buy')
    plt.scatter(close_sells, df["Close"].iloc[close_sells], color='purple', label='close Sell')

    # Set labels and title
    plt.xlabel("Time")
    plt.ylabel("Closed Price")
    plt.title("Closed Price Chart with Buy/Sell Points")

    # Display the legend
    plt.legend()

    # Show the plot
    plt.show()
    return 0


if __name__ == "__main__":
    main()