import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import telebot
import time
import io 
import numpy as np
from PIL import Image
from binance.um_futures import UMFutures
from datetime import timedelta
from tinkoff.invest import Client
from tinkoff.invest.utils import now, quotation_to_decimal
from tinkoff.invest.services import InstrumentsService

TINKOFF_TOKEN = os.environ["TINKOFF_API"]
TELEGAM_TOKEN = os.environ["TELEGRAM_API"]
CHAT_TOKEN = os.environ["CHAT_TOKEN"]

pd.set_option('display.max_rows', None)


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
    def from_tinkoff(self, figi = 'BBG004730N88', interval = 5, hours = 365) -> pd.DataFrame:
        with Client(TINKOFF_TOKEN) as client:
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
        plt.figure('prices', frameon=False, figsize=(16,3))
        plt.plot(self.df['Close'])
        if 'EMA1' in self.df:
            plt.plot(self.df['EMA1'])
            plt.plot(self.df['UP'])
            plt.plot(self.df['DOWN'])
        if 'EMA2' in self.df:
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
        if 'OBV' in self.df:
            plt.figure('OBV', figsize=(16,3))
            plt.plot(self.df['OBV'])
        plt.show()
    
    def create_img(self):
        img = None
        plt.figure('prices', frameon=False, figsize=(16,9))
        width = .8
        width2 = .1
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
        if 'Signals' in self.df:
            for i in range(len(self.df['Signals'])):
                if self.df['Signals'].iloc[i] == 1:
                    plt.axvline(x=self.df.index[i], color='green',linewidth=2)
                elif self.df['Signals'].iloc[i] == -1:
                    plt.axvline(x=self.df.index[i], color='red',linewidth=2)
        fig = plt.gcf()
        buf = io.BytesIO() 
        fig.savefig(buf) 
        plt.clf()
        buf.seek(0) 
        img = Image.open(buf) 
        return img 
    
    def add_EMA1(self, len):
        self.df['EMA1'] = ta.ema(self.df['Close'], len)

    def add_EMA2(self, len):
        self.df['EMA2'] = ta.ema(self.df['Close'], len)

    def add_RSI(self, len):
        self.df['RSI'] = ta.rsi(self.df['Close'], len)

    def add_OBV(self):
        self.df['OBV'] = ta.obv(self.df['Close'], self.df['Volume'])
         
    def train(self, len_s):
        df = self.df.dropna()
        price = np.array(df['Close'].tolist(), float)
        price_norm = (price - price. min ())/ (price. max () - price. min ())
        ema1 = np.array(df['EMA1'].tolist(), float)
        ema1_norm = (ema1 - ema1. min ())/ (ema1. max () - ema1. min ())
        ema2 = np.array(df['EMA2'].tolist(), float)
        ema2_norm = (ema2 - ema2. min ())/ (ema2. max () - ema2. min ())
        obv = np.array(df['OBV'].tolist(), float)
        obv_norm = (obv - obv. min ())/ (obv. max () - obv. min ())
        rsi = np.array(df['RSI'].tolist(), float)
        rsi_norm = (rsi - rsi. min ())/ (rsi. max () - rsi. min ())
        all_profit = 0

        while all_profit <= 0:

            w0 = np.random.rand(10,len_s)
            w1 = np.random.rand(len_s,3)

            signals = [0] * len(price)
            buy_val = 0
            all_profit = 0

            for k in range(len(price)):

                s = np.array([0] * len_s, float)
                y = np.array([0] * 3, float)
                x = np.array([
                    price_norm[k-1],
                    price_norm[k],
                    ema1_norm[k-1],
                    ema1_norm[k],
                    ema2_norm[k-1],
                    ema2_norm[k],
                    obv_norm[k-1],
                    obv_norm[k],
                    rsi_norm[k-1],
                    rsi_norm[k]
                ], float)
                for i in range(len_s):
                    for j in range(10):
                        s[i] += x[j] * w0[j][i]
                for i in range(3):
                    for j in range(len_s):
                        y[i] += s[j] * w1[j][i]

                n = np.argmax(y)

                if n == 1:
                    signals[i] = 1
                    buy_val = 1000 / price[i]
                elif n == 2:
                    signals[i] = -1
                    all_profit += ((buy_val * price[i] - 1000) / 1000) - 0.006
                    buy_val = 0
            print(all_profit)

    def add_signals(self, set1, set2, set3, set4):
        self.add_EMA1(set1, set2) #1-10, #0.02 - 0.15
        self.add_EMA2(set3) #5 1-10
        self.add_RSI(set4,2) #2-10
        self.add_OBV()
        
        

        price = self.df['Close'].tolist()
        ema1 = self.df['EMA1'].tolist()
        ema2 = self.df['EMA2'].tolist()
        obv = self.df['OBV'].tolist()
        rsi = self.df['RSI'].tolist()
        up = self.df['UP'].tolist()
        down = self.df['DOWN'].tolist()
        signals = [0] * len(price)
        buy_val = 0
        all_profit = 0

        self.df['Signals'] = pd.Series(signals, index=self.df.index)
        return all_profit

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

def find_shares():
    shares_df = get_shares_list()
    df = pd.DataFrame(columns=['name', 'figi', 'ticker', 'rs'])
    for i in range(len(shares_df['figi'])):
        time.sleep(0.2)
        share = Candle(shares_df['figi'].iloc[i], '1d', limit=1000)
        r = 0
        rs = 0
        for q in range(1,10):
            test = share
            m = test.add_signals(q)
            if m > r:
                r = m
                rs = q
            test = None
        
        if r > 0.3:
            print(shares_df['name'].iloc[i], r, rs)
            df.loc[ len(df.index )] = [shares_df['name'].iloc[i],shares_df['figi'].iloc[i], shares_df['ticker'].iloc[i], rs]
            
    df.to_csv('list.csv', sep=';', index=False)

def send_signals():
    bot = telebot.TeleBot(TELEGAM_TOKEN)
    df = pd.read_csv("list.csv", sep=';')
    for i in range(len(df)):
        share = Candle(df['figi'].iloc[i],'1d', limit=365)
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
    a = Candle("BTCUSDT", limit=100)
    a.add_EMA1(4)
    a.add_EMA2(14)
    a.add_OBV()
    a.add_RSI(13)
    a.train(20)
    return 0

if __name__ == "__main__":
    main()