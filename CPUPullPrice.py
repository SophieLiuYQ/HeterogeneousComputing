## this file is the same as test2.py in data folder. 
## the code runs on a remote server to pull price.

import data_processing
import datetime
import os
import pandas as pd
import yfinance as yf

from bs4 import BeautifulSoup as BS
from pandas_market_calendars import get_calendar
from zoneinfo import ZoneInfo

now = datetime.datetime.now()
now = now.astimezone(ZoneInfo('US/Eastern'))

timedelta = datetime.timedelta(days=1)

mcal_nyse = get_calendar('NYSE')
schedule = mcal_nyse.schedule((now - timedelta).strftime('%Y-%m-%d'),
                              (now + timedelta).strftime('%Y-%m-%d'))

now = now.strftime("%Y-%m-%dT%H:%M:%S%z")
market_open = mcal_nyse.open_at_time(schedule, now)

tickers = data_processing.read_portfolio()

for ticker in tickers:
    filepath = os.path.join(
        data_processing._DEFAULT_EXPORT_DIR, f'price_{ticker}.csv')
    print('fetching price for', ticker, 'to update', filepath)
    ticker = yf.Ticker(ticker)
    data = [(ticker.info['regularMarketPrice'], now, market_open)]
    df = pd.DataFrame(data, columns=['price','time','market_open'])
    try:
        read_prices = lambda: pd.read_csv(
            filepath, dtype={"price": "float", "time": "str", "market_open": "bool"})
        df = pd.concat([read_prices(), df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(filepath, index=False)
