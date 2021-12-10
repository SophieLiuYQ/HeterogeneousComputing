from bs4 import BeautifulSoup as BS

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

import pandas as pd
import sys, time
from datetime import datetime

datetime.now().strftime('%Y-%m-%d %H:%M:%S')

csv_format = "data_%s.csv"
exe = "/Downloads"
url = "https://finance.yahoo.com/quote/{0}/news?p={0}"
tickers = ['FB','MSFT','AAPL','NVDA','AMD','XLNX','QCOM','MU']
## tickers = ["INTC"]
service = Service(exe)


def is_advertisement(li) -> bool:
    return any(a.getText() == "Ad" for a in li.find_all("a"))

def req_with_scrolling(url, max_scroll):
    ##### Web scrapper for infinite scrolling page #####
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    sleep_time = 1
    time.sleep(sleep_time)  # Allow 2 seconds for the web page to open
    offset = driver.execute_script("return window.pageYOffset;")
    height = driver.execute_script("return window.screen.height;")
    last_offset, i = (-1, 1)

    while (offset != last_offset) and (i < max_scroll):
        last_offset, i = (offset, i + 1)
        driver.execute_script("window.scrollTo(0, %d);" % (i * height))
        time.sleep(sleep_time / 2)  # Allow 2 seconds for the web page to open
        offset = driver.execute_script("return window.pageYOffset;")

    page_source = driver.page_source
    driver.quit()
    return page_source


def req_ticker(ticker, max_scroll=sys.maxsize):
    ticker_url = url.format(ticker)
    print("info> reading:", ticker_url)
    soup = BS(req_with_scrolling(ticker_url, max_scroll), "html.parser")
    lis = soup.find_all("li", {"class": "js-stream-content Pos(r)"})
    h3s = sum([li.find_all("h3") for li in lis if not is_advertisement(li)], [])
    return [h3.getText() for h3 in h3s]

now = datetime.now().strftime('%m/%d/%y %H:%M')


for ticker in tickers:
    csv = csv_format % ticker
    headings = req_ticker(ticker, 8)
    data = list(zip(headings, [now] * len(headings)))
    tmp_df = pd.DataFrame(reversed(data), columns=[ticker,'time'])
    try:
        csv_df = pd.read_csv(csv)
        df = pd.concat([csv_df, tmp_df], ignore_index=True)
    except:
        df = tmp_df
    df = df.drop_duplicates(subset=[ticker], keep="first", ignore_index=True)
    df.to_csv(csv, index=False)
