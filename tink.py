import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
import telebot
import io
import math
import time
import datetime
from binance.um_futures import UMFutures
from datetime import timedelta, timezone
from PIL import Image
from tinkoff.invest import CandleInterval, Client, PortfolioResponse, PortfolioPosition, AccessLevel, RequestError, OrderDirection, OrderType, Quotation
from tinkoff.invest.utils import now, quotation_to_decimal
from tinkoff.invest.services import InstrumentsService, MarketDataService, Services

TOKEN = os.environ["MAIN_TOKEN"]
token = os.environ["TELEGRAM_TOKEN"]
chat_id = os.environ["CHAT_ID"]
admin_chat_id = os.environ["ADMIN_CHAT_ID"]
account_id = os.environ["ACCOUNT_ID"]
api_key = os.environ["BINANCE_API_KEY"]
sec_api = os.environ["BINANCE_SEC_KEY"]

bot = telebot.TeleBot(token)

money_in_order = 100

h_calc = 10
thld = 0.01

can_short = True

class Portfolio:
    def __init__(self, client: Services):
        self.client = client
        self.accounts = []

    def get_currency(self, valut = 'rub'):
        if valut == 'usd':
            symbol = 'BBG0013HGFT4'
        elif valut == 'hkd':
            symbol = 'BBG0013HSW87'
        elif valut == 'rub':
            symbol = 'RUB000UTSTOM'

        dataframes = []
        for account_id in self.get_accounts():
            df = self.get_portfolio_df(account_id)
            if df is None: continue
            dataframes.append(df)
        df = pd.concat(dataframes, ignore_index=True)
        df = df.query("instrument_type == 'currency'")
        df = df[df['figi'] == symbol]
        df = df.reset_index(drop=True)
        return df

    def report(self):
        dataframes = []
        for account_id in self.get_accounts():
            df = self.get_portfolio_df(account_id)
            if df is None: continue
            dataframes.append(df)
        df = pd.concat(dataframes, ignore_index=True)
        df = df.query("instrument_type == 'share'")
        df = df.reset_index(drop=True)
        return df

    def get_accounts(self):
        r = self.client.users.get_accounts()
        for acc in r.accounts:
            if acc.access_level != AccessLevel.ACCOUNT_ACCESS_LEVEL_NO_ACCESS:
                self.accounts.append(acc.id)

        return self.accounts


    def get_portfolio_df(self, account_id : str):
        r: PortfolioResponse = self.client.operations.get_portfolio(account_id=account_id)
        if len(r.positions) < 1: return None
        df = pd.DataFrame([self.portfolio_pose_todict(p) for p in r.positions])
        return df

    def portfolio_pose_todict(self, p : PortfolioPosition):
        r = {
            'figi': p.figi,
            'quantity': quotation_to_decimal(p.quantity),
            'current_price': float(quotation_to_decimal(p.current_price)),
            'currency': p.average_position_price.currency,
            'instrument_type': p.instrument_type,
            'quantity_lot' : quotation_to_decimal(p.quantity_lots)
        }
        return r

def get_shares_list(): #готово
    with Client(TOKEN) as cl:
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
        df = df.query("currency == 'rub' and qual_flag == False and api_flag == True")
        df = df.reset_index(drop=True)
        return df

def get_ticker_and_name(symbol):
    with Client(TOKEN) as cl:
        instruments: InstrumentsService = cl.instruments
        market_data: MarketDataService = cl.market_data
 
        l = []
        for method in ['shares']:#, 'bonds', 'etfs', 'currencies', 'futures']:
            for item in getattr(instruments, method)().instruments:
                l.append({
                    'ticker': item.ticker,
                    'figi': item.figi,
                    'type': method,
                    'name': item.name,
                })
 
        df = pd.DataFrame(l)
        df = df[df['figi'] == symbol]
 
        return df['ticker'].iloc[0], df['name'].iloc[0]

def get_figi_info(symbol, i = True):
    with Client(TOKEN) as cl:
        instruments: InstrumentsService = cl.instruments
        market_data: MarketDataService = cl.market_data
 
        l = []
        for method in ['shares', 'currencies', 'bonds', 'etfs', 'futures']:
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
        if i:
            df = df[df['figi'] == symbol]
        else:
            df = df[df['ticker'] == symbol]
        df = df.reset_index(drop=True)
 
        return df

def Supertrend(df, atr_period = 10, multiplier = 3.0):
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    price_diffs = [high - low, 
                   high - close.shift(), 
                   close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    atr = true_range.ewm(alpha=1/atr_period,min_periods=atr_period).mean() 
    
    hl2 = (high + low) / 2
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)
    
    supertrend = [True] * len(df)
    
    for i in range(1, len(df.index)):
        curr, prev = i, i-1
        
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        else:
            supertrend[curr] = supertrend[prev]
            
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan
    
    return pd.DataFrame({
        'st': supertrend,
        'lower': final_lowerband,
        'upper': final_upperband
    }, index=df.index)

def get_data_bin(symbol, limit = 365, interval = '1d'):
    um_futures_client = UMFutures()
    res = um_futures_client.klines(symbol, interval, limit = limit)
    data = pd.DataFrame(res, columns=['time', 'open', 'high', 'low', 'close', 'volume', '1', '2', '3', '4', '5', '6'])
    data = data.drop(['1', '2', '3', '4', '5', '6'], axis=1)
    for i in range(len(data['time'])):
        data.loc[i , 'time'] = pd.to_datetime(data.loc[i , 'time'], utc=True, unit='ms')
        data.loc[i , 'open'] = float(data.loc[i , 'open'])
        data.loc[i , 'high'] = float(data.loc[i , 'high'])
        data.loc[i , 'low'] = float(data.loc[i , 'low'])
        data.loc[i , 'close'] = float(data.loc[i , 'close'])
    print(data)
    return data

def get_data(figi, days = 365, interval = CandleInterval.CANDLE_INTERVAL_DAY): #готово
    data = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    with Client(TOKEN) as client:
        for candle in client.get_all_candles(
            figi = figi,
            from_ = now() - timedelta(days=days),
            interval=interval,
        ):
            data1 = pd.DataFrame({'time': [candle.time],
                                  'open': [float(quotation_to_decimal(candle.open))],
                                  'high': [float(quotation_to_decimal(candle.high))],
                                  'low': [float(quotation_to_decimal(candle.low))],
                                  'close': [float(quotation_to_decimal(candle.close))],
                                  'volume': [candle.volume]})
            data = pd.concat([data, data1], ignore_index=True)
    
    return data

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, )
    buf.seek(0)
    img = Image.open(buf)
    return img

def send_plot(data, chat = chat_id):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(data['close'])
    axs[0].plot(data['EMA'])
    #axs[0].plot(data['lower'], color='g')
    #axs[0].plot(data['upper'], color='r')
    axs[1].plot(data['rsi'])
    axs[1].plot(data['rsiEMA'])
    axs[1].plot([30] * len(data['rsi']))
    axs[1].plot([70] * len(data['rsi']))
    axs[0].set_xlabel(str(data['profit'].iloc[-1]))
    
    for i in range(len(data['orders'])):
        if data['orders'][i] == 1:
            axs[0].axvline(x=i, color='green',linewidth=2)
        elif data['orders'][i] == -1:
            axs[0].axvline(x=i, color='red',linewidth=2)

    plt.show()
    fig = plt.gcf()
    img = fig2img(fig)
    bot.send_photo(chat, img)
    plt.close()

def to_long(data, n):
    return (data['EMAsmall'].loc[n] > data['EMAmax'].loc[n] and data['EMAsmall'].loc[n-1] < data['EMAmax'].loc[n-1])

def to_short(data, n):
    return (data['EMAsmall'].loc[n] < data['EMAmin'].loc[n] and data['EMAsmall'].loc[n-1] > data['EMAmin'].loc[n-1])

def to_close_long(data, n):
    return (data['EMAsmall'].loc[n] < data['EMAmax'].loc[n] and data['EMAsmall'].loc[n-1] > data['EMAmax'].loc[n-1]) or (to_short(data, n))

def to_close_short(data, n):
    return (data['EMAsmall'].loc[n] > data['EMAmin'].loc[n] and data['EMAsmall'].loc[n-1] < data['EMAmin'].loc[n-1]) or (to_long(data, n))

def calc_ind(data):
    data = data.join(Supertrend(data))
    data['rsi'] = ta.rsi(data['close'],)
    data['rsiEMA'] = ta.ema(data['rsi'], 20)
    data = data.join(back_test(data))
    return data

def search_shares():
    df = get_shares_list()
    len_df = len(df['figi'])

    for i in range(len_df):
        figi = df['figi'][i]
        data = get_data(figi)
        data = calc_ind(data)
        if data.empty == False:
            if to_long(data, len(data['close']) - 1) or to_long(data, len(data['close']) - 2) or to_long(data, len(data['close']) - 3) or to_long(data, len(data['close']) - 4) or to_long(data, len(data['close']) - 5):
                ticker = df['ticker'][i]
                name = df['name'][i]
                currency = df['currency'][i]
                price = data['close'].iloc[-1]
                profit = data['profit'].iloc[-1]

                bot.send_message(chat_id,
                                f'Можно покупать `{ticker}`\nназвание [{name}](https://www.tinkoff.ru/invest/stocks/{ticker})\nцена: {price} {currency}\nпрофит: {profit}',
                                parse_mode='MARKDOWN') 
                send_plot(data)

        print(round(i / len_df * 100, 2), '%')

def update():
    with Client(TOKEN) as client:
        sh_list = Portfolio(client).report()
    len_df = len(sh_list['figi'])

    for i in range(len_df):
        figi = sh_list['figi'][i]
        data = get_data(figi)
        data = calc_ind(data)      
        if to_short(data, len(data['close']) -1 ):
            ticker, name = get_ticker_and_name(figi)
            price = sh_list['current_price'][i]
            currency = sh_list['currency'][i]
            profit = data['profit'].iloc[-1]
            bot.send_message(chat_id,
                            f'Пора продавать  `{ticker}`\nназвание [{name}](https://www.tinkoff.ru/invest/stocks/{ticker})\nцена: {price} {currency}\nпрофит: {profit}',
                            parse_mode='MARKDOWN') 
            send_plot(data)
        print(round(i / len_df * 100, 2), '%')

def back_test(data, set = False):
    order_vol = 100
    order_limit = 5

    orders_list = [0]
    profit_list = [0]
    profit = 0
    long_stack = []
    short_stack = []
    

    for i in range(1, len(data['close'])):
        all_price = 0
        quantity = 0
        count = 0 
        current_price = data['close'][i]
        orders = 0

        if to_close_long(data, i):
            if long_stack:
                for price in long_stack:
                    count += 1
                    all_price += price
                    quantity += order_vol / price
                avg_price_entry = all_price / count
                profit += (current_price - avg_price_entry) * quantity
                profit -= 0.04 * count
                long_stack.clear()
                orders -= 1
        
        elif to_close_short(data, i):
            if short_stack:
                for price in short_stack:
                    count += 1
                    all_price += price
                    quantity += order_vol / price
                avg_price_entry = all_price / count
                profit += (avg_price_entry - current_price) * quantity
                profit -= 0.04 * count
                short_stack.clear()
                orders += 1

        if to_long(data, i):                
            if len(long_stack) < order_limit:
                profit -= 0.04
                long_stack.append(current_price)
                orders += 1

        elif to_short(data, i):
            if can_short and len(short_stack) < order_limit:
                profit -= 0.04
                short_stack.append(current_price)
                orders -= 1

        orders_list.append(orders)
        profit_list.append(profit)
            
    return pd.DataFrame({
            'orders': orders_list,
            'profit': profit_list
        }, index=data.index)

def order(figi, dir, quentity = 0):
    if dir == 1:
        dir_ord = OrderDirection.ORDER_DIRECTION_BUY
    elif dir == -1:
        dir_ord = OrderDirection.ORDER_DIRECTION_SELL
    
    res = get_figi_info(figi)
    currency = res['currency'][0]
    lot = res['quantity_lot'][0]
    price = get_data(figi, 1)['close'][0]
    with Client(TOKEN) as client:
        res = Portfolio(client).get_currency(currency)
        value = res['quantity'][0]
        figi_val = res['figi'][0]
        val_price = 1
    if currency != 'rub':
        val_price = get_data(figi_val, 1)['close'][0]

    if quentity:
        quent = quentity
    else:
        if price * lot * val_price > money_in_order:
            quent = 1
        else:
            quent = int(money_in_order / (price * lot * val_price))
    try:        
        if currency != 'rub' and value < quent * price * lot and dir == 1:
            with Client(TOKEN) as client:
                figi_val = Portfolio(client).get_currency(currency)['figi'][0]
                r = client.orders.post_order(
                        order_id=str(datetime.utcnow().timestamp()),
                        figi=figi_val,
                        quantity=math.ceil(quent * price * lot),
                        account_id=account_id,
                        direction=dir_ord,
                        order_type=OrderType.ORDER_TYPE_MARKET
                    )
                
        with Client(TOKEN) as client:
            r = client.orders.post_order(
                    order_id=str(datetime.utcnow().timestamp()),
                    figi=figi,
                    quantity=quent,
                    account_id=account_id,
                    direction=dir_ord,
                    order_type=OrderType.ORDER_TYPE_MARKET
                    )
    except Exception:
        bot.send_message(admin_chat_id, 'Недостаточно средств или биржа закрыта')

def full_acc_backtest():
    with Client(TOKEN) as client:
        sh_list = Portfolio(client).report()
    len_df = len(sh_list['figi'])
    for i in range(len_df):
        figi = sh_list['figi'][i]
        data = get_data(figi)
        data = calc_ind(data)
        ticker, name = get_ticker_and_name(figi)
        bot.send_message(admin_chat_id, f'{ticker}\n{name}')
        send_plot(data, admin_chat_id)      

def main():    #event, context
    import gym
    import gym_anytrading
    from gym_anytrading.envs import StocksEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import A2C

    pd.set_option('display.max_rows', None) 
    data = get_data_bin('BTCUSDT', 30, '1d')
    df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(data['time'])
    df['Open'] = pd.to_numeric(data['open'], errors='coerce')
    df['High'] = pd.to_numeric(data['high'], errors='coerce')
    df['Low'] = pd.to_numeric(data['low'], errors='coerce')
    df['Close'] = pd.to_numeric(data['close'], errors='coerce')
    df['Volume'] = pd.to_numeric(data['volume'], errors='coerce')
    df['RSI'] = ta.rsi(df['Close'],10)
    df['EMA'] = ta.ema(df['Close'],10)
    df['OBV'] = ta.obv(df['Close'],df['Volume'])
    df.set_index('Date', inplace=True)
    print(df)
    def add_signals(env):
        start = env.frame_bound[0]
        end = env.frame_bound[1]
        prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
        signal_features = env.df.loc[:, ['Open', 'High', 'Low', 'Volume','EMA', 'RSI', 'OBV']].to_numpy()[start:end]
        return prices, signal_features
    
    class MyCustomEnv(StocksEnv):
        _process_data = add_signals
        
    env2 = MyCustomEnv(df=df, window_size=12, frame_bound=(12,1000))
    env_maker = lambda: env2
    env = DummyVecEnv([env_maker])
    model = A2C('MlpPolicy', env, verbose=2)
    model.learn(total_timesteps=100000)

    print(df.tail(20))

    env = MyCustomEnv(df=df, window_size=12, frame_bound=(1100, 1500))
    obs = env.reset()
    while True:
        obs = obs[np.newaxis, ...]
        action, _state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            print(info)
            break
    plt.figure(figsize=(15,6))
    plt.cla()
    env.render_all()
    plt.show()
    return 0

if __name__ == "__main__":
    main()