## this file is the same as test2.py in data folder. 
## the code runs on a remote server to pull price.

from bs4 import BeautifulSoup as BS

import requests
import pandas as pd
import sys, time
from datetime import datetime

url = "https://finance.yahoo.com/quote/{0}/news?p={0}"
tickers = ['FB','MSFT','AAPL','NVDA','AMD','XLNX','QCOM','MU']
##tickers = ['INTC']
csv_format = "price_%s.csv"

def req_ticker(ticker):
    ticker_url = url.format(ticker)
    print("info> reading:", ticker_url)
    req = requests.get(ticker_url, headers= {'User-agent': 'Custom'})
    soup = BS(req.text,"html.parser")
    if (req.status_code != 200):
        print("warning> got status code %d\n", req.status_code, file=sys.stderr)
        print(soup.prettify(), file=sys.stderr)
    lis = soup.find_all('fin-streamer', {"data-field": "regularMarketPrice", "data-symbol": ticker})
    return [ li.getText() for li in lis ]

now = datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')

for ticker in tickers:
    csv = csv_format % ticker
    headings = req_ticker(ticker)
    data = list(zip(headings, [now] * len(headings)))
    tmp_df = pd.DataFrame(reversed(data), columns=[ticker,'time'])
    try:
        csv_df = pd.read_csv(csv)
        df = pd.concat([csv_df, tmp_df], ignore_index=True)
    except:
        df = tmp_df
    ##df = df.drop_duplicates(subset=[ticker], keep="first", ignore_index=True)
    df.to_csv(csv, index=False)
