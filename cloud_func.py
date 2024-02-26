import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import telebot
import time
import io 
import numpy as np
import random
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
            plt.plot(self.df['RSI'].index.values,[30]*len(self.df['RSI']))
        if 'OBV' in self.df:
            plt.figure('OBV', figsize=(16,3))
            plt.plot(self.df['OBV'])
        plt.show()
    
    def create_img(self):
        img = None
        plt.figure('prices', frameon=False, figsize=(16,9))
        plt.plot(self.df['Close'])
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

    def add_signals(self):
        price = self.df['Close'].tolist()
        ema1 = self.df['EMA1'].tolist()
        obv = self.df['OBV'].tolist()
        rsi = self.df['RSI'].tolist()
        signals = [0] * len(price)

        for i in range(len(price)):
            if rsi[i] >= 30 and rsi[i-1] < 30 and (obv[i] > obv[i-1] or obv[i-1] > obv[i-2]):
                signals[i] = 1

        self.df['Signals'] = pd.Series(signals, index=self.df.index)

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
        df = df.query("currency in ('rub') and qual_flag == False and api_flag == True")
        df = df.reset_index(drop=True)
        return df

def send_signals():
    bot = telebot.TeleBot(TELEGAM_TOKEN)
    df = get_shares_list()
    for i in range(len(df)):
        share = Candle(df['figi'].iloc[i])
        share.add_EMA1(4)
        share.add_RSI(20)
        share.add_OBV()
        share.add_signals()
        if share.df['Signals'].iloc[-1] == 1:
            name = df['name'].iloc[i]
            ticker = df['ticker'].iloc[i]
            msg = f'1Покупка\n[{name}](https://www.tinkoff.ru/invest/stocks/{ticker})'
            img = share.create_img()
            bot.send_photo(CHAT_TOKEN, img)
            bot.send_message(CHAT_TOKEN, msg, parse_mode='Markdown')
            time.sleep(5)
            
            
        elif share.df['Signals'].iloc[-1] == -1:
            name = df['name'].iloc[i]
            ticker = df['ticker'].iloc[i]
            msg = f'2Продажа\n[{name}](https://www.tinkoff.ru/invest/stocks/{ticker})'
            img = share.create_img()
            bot.send_photo(CHAT_TOKEN, img)
            bot.send_message(CHAT_TOKEN, msg, parse_mode='Markdown')
            time.sleep(5)
        share.draw()


def main():
    send_signals()
    return 0

if __name__ == "__main__":
    main()