import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import telebot
import time
import numpy as np
import copy
from binance.um_futures import UMFutures
import datetime
from tinkoff.invest import Client
from tinkoff.invest.utils import now, quotation_to_decimal
from tinkoff.invest.services import InstrumentsService

TINKOFF_TOKEN = os.environ["TINKOFF_API"]
TELEGAM_TOKEN = os.environ["TELEGRAM_API"]
CHAT_TOKEN = os.environ["CHAT_TOKEN"]

pd.set_option('display.max_rows', None)


class Price_chart:

    """Create Pandas DataFrame from Binance or Tinkoff"""

    def __init__(self, symbol:str, interval='1d', limit=365, start_time = '') -> None:
        assert interval in ['1m', '5m', '15m', '1h', '1d']
        self.df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        if len(symbol) == 12 or len(symbol) == 4:
            match interval:
                case '1m':
                    interval = 1
                    mult = 1 / 60
                case '5m':
                    interval = 2
                    mult = 1 / 60 * 5
                case '15m':
                    interval = 3
                    mult = 1 / 60 * 15
                case '1h':
                    interval = 4
                    mult = 1
                case '1d':
                    interval = 5
                    mult = 24
            if len(symbol) == 4:
                symbol = self.get_figi(symbol)
            self.from_tinkoff(symbol, interval, limit * mult, start_time)
        elif 4 < len(symbol) < 12:
            if start_time != '':
                bin_start = datetime.datetime.strptime(start_time, "%Y-%m-%d").date()
                bin_start = datetime.datetime.combine(bin_start, datetime.datetime.min.time())
                bin_start = int(bin_start.timestamp()*1000)
                self.from_binance(symbol, interval, limit, bin_start)
            else:
                self.from_binance(symbol, interval, limit, start_time)

    def from_binance(self, symbol= 'BTCUSDT', interval='1d', limit=365, start_time = '') -> pd.DataFrame:
        um_futures_client = UMFutures()
        if start_time == '':
            res = um_futures_client.klines(symbol=symbol, interval=interval, limit=limit)
        else:
            res = um_futures_client.klines(symbol=symbol, interval=interval, limit=limit, startTime=start_time)
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
        with Client(TINKOFF_TOKEN) as cl:
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
        
    def from_tinkoff(self, figi = 'BBG004730N88', interval = 5, hours = 365, start_time = 0) -> pd.DataFrame:
        if start_time == 0:
            start = now() - datetime.timedelta(hours=hours)
        else:
            start = start_time
        with Client(TINKOFF_TOKEN) as client:
            for candle in client.get_all_candles(
                figi = figi,
                from_ = start,
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
        plt.figure('prices', frameon=False, figsize=(16,9))
        match self.interval:
                case '1m':
                    mult = 24*60/1
                case '5m':
                    mult = 24*60/5
                case '15m':
                    mult = 24*60/15
                case '1h':
                    mult = 24
                case '1d':
                    mult = 1
        width = .8 / mult
        width2 = .1 / mult
        up = self.df[self.df. Close >=self.df. Open ]
        down = self.df[self.df. Close <self.df. Open ]
        col1 = 'green'
        col2 = 'red'
        plt.bar (up. index ,up. Close -up. Open ,width,bottom=up. Open ,color=col1)
        plt.bar (up. index ,up. High -up. Close ,width2,bottom=up. Close ,color=col1)
        plt.bar (up. index ,up. Low -up. Open ,width2,bottom=up. Open ,color=col1)
        plt.bar (down. index ,down. Close -down. Open ,width,bottom=down. Open ,color=col2)
        plt.bar (down. index ,down. High -down. Open ,width2,bottom=down. Open ,color=col2)
        plt.bar (down. index ,down. Low -down. Close ,width2,bottom=down. Close ,color=col2)
        plt.xticks (rotation= 45 , ha='right')
        plt.grid(True)
        plt.show() 
    
class test_Trade(Price_chart):

    def __init__(self, symbol:str, interval='1d', limit=365, start_time = '') -> None:
        super(test_Trade, self).__init__(symbol, interval, limit, start_time)
        self.money = 10000
        self.quantity = 0
        self.num_money = 4000

    def buy(self, price):
        if self.money - self.num_money > 0:
            self.quantity += (self.num_money / price) - (self.num_money / price * 0.005)
            self.money -= self.num_money
            return True
        else:
            return False

    def sell(self, price):
        if self.quantity > 0:  
            self.money += (self.quantity * price) - (self.quantity * price * 0.005)
            self.quantity = 0
            return True
        else:
            return False

    def add_borders(self, len_EMA1, len_EMA2):
        self.df['EMA1'] = ta.ema(self.df['Close'], len_EMA1)
        self.df['EMA2'] = ta.ema(self.df['Close'], len_EMA2)
        
    def add_RSI(self, len_RSI, len_EMA):
        RSI = ta.rsi(self.df['Close'], len_RSI)
        self.df['RSI'] = ta.ema(RSI, len_EMA)

    def add_OBV(self, len_EMA):
        OBV = ta.obv(self.df['Close'], self.df['Volume'])
        self.df['OBV'] = ta.ema(OBV, len_EMA)

    def add_ATR(self, len_ATR, len_EMA):
        ATR = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], len_ATR)
        self.df['ATR'] = ta.ema(ATR, len_EMA)

    def draw_all(self):
        plt.figure('prices', frameon=False, figsize=(16,3))
        plt.plot(self.df['Close'])
        plt.grid(True)
        if 'EMA1' in self.df:
            plt.plot(self.df['EMA1'])
            plt.plot(self.df['EMA2'])
        if 'Signals' in self.df:
            for i in range(len(self.df['Signals'])):
                if self.df['Signals'].iloc[i] == 1:
                    plt.axvline(x=self.df.index[i], color='green',linewidth=2)
                elif self.df['Signals'].iloc[i] == -1:
                    plt.axvline(x=self.df.index[i], color='red',linewidth=2)
        if 'RSI' in self.df:
            plt.figure('RSI', figsize=(16,3), )
            plt.plot(self.df['RSI'])
            plt.plot(self.df['RSI'].index.values,[50]*len(self.df['RSI']))
            plt.grid(True)
        if 'OBV' in self.df:
            plt.figure('OBV', figsize=(16,3))
            plt.plot(self.df['OBV'])
            plt.grid(True)
        if 'ATR' in self.df:
            plt.figure('ATR', figsize=(16,3))
            plt.plot(self.df['ATR'])
            plt.grid(True)
        plt.show()

def train(df, set1, set2):
    df.add_borders(set1, set2)
    
    price = df.df['Close'].tolist()
    ema1 = df.df['EMA1'].tolist()
    ema2 = df.df['EMA2'].tolist()
    signals = [0] * len(price)

    for i in range(set1, len(price)):
        if ema1[i] > ema2[i]:
            if df.buy(price[i]):
                signals[i] = 1
        elif ema1[i] < ema2[i]:
            if df.sell(price[i]):
                signals[i] = -1
        
    df.df['Signals'] = signals
    #df.sell(df.df['Close'].iloc[-1])
    return df

def get_shares_list(): #готово
    with Client(TINKOFF_TOKEN) as cl:
        instruments: InstrumentsService = cl.instruments
        l = []
        for method in ['shares']:#, 'bonds', 'etfs', 'currencies', 'futures']:
            for item in getattr(instruments, method)().instruments:
                l.append({
                    'ticker': item.ticker,
                    'figi': item.figi,
                    'name': item.name,
                    'currency': item.currency,
                    'quantity_lot': item.lot,
                    'buy_flag': item.buy_available_flag,
                    'sell_flag': item.sell_available_flag,
                    'api_flag': item.api_trade_available_flag,
                    'qual_flag': item.for_qual_investor_flag
                })
        df = pd.DataFrame(l)
        df = df.query("currency in ('rub','hkd') and qual_flag == False and api_flag == True")
        df = df.reset_index(drop=True)
        return df

def send_signals():
    bot = telebot.TeleBot(TELEGAM_TOKEN)
    df = pd.read_csv("list.csv", sep=';')
    for i in range(len(df)):
        share = Price_chart(df['figi'].iloc[i],'1d', limit=365)
        r= share.add_signals(df['rs'].iloc[i])
        if share.df['Signals'].iloc[-1] == 1:
            name = df['name'].iloc[i]
            ticker = df['ticker'].iloc[i]
            msg = f'1Покупка\nГодовой профит: {round(r*100)}%\n[{name}](https://www.tinkoff.ru/invest/stocks/{ticker})'
            img = share.create_img()
            bot.send_photo(CHAT_TOKEN, img)
            bot.send_message(CHAT_TOKEN, msg, parse_mode='Markdown')
            time.sleep(5)
            
        elif share.df['Signals'].iloc[-1] == -1:
            name = df['name'].iloc[i]
            ticker = df['ticker'].iloc[i]
            msg = f'2Продажа\nГодовой профит: {round(r*100)}%\n[{name}](https://www.tinkoff.ru/invest/stocks/{ticker})'
            img = share.create_img()
            bot.send_photo(CHAT_TOKEN, img)
            bot.send_message(CHAT_TOKEN, msg, parse_mode='Markdown')
            time.sleep(5)



def main():
    a = test_Trade("BTCUSDT", '1h', 1000, '2020-01-01')
    d = copy.deepcopy(a)
    best_i = 0
    best_j = 0
    money = 10000
    for i in range(1,100):
        for j in range(2,200,2):
            c = copy.deepcopy(a)
            train(c, i, j)
            if c.money > money:
                best_i = i
                best_j = j
                money = c.money
                d = copy.deepcopy(c)
            print(money, best_i, best_j)
    d.draw_all()
    return 0

if __name__ == "__main__":
    main()