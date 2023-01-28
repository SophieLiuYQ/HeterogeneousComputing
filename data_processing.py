import csv
import datetime
import logging
import os
import pandas as pd
import pandas_market_calendars
import yfinance as yf
import zoneinfo

try:
    import microquant
except ImportError:
    microquant = None

ZoneInfo = zoneinfo.ZoneInfo

_DEFAULT_API_KEY_FILE = "keystore.csv"
_DEFAULT_EXPORT_DIR = "data/"
_DEFAULT_NEWS_PREFIX = "data"
_DEFAULT_PORTFOLIO_FILE = "portfolio.txt"
_DEFAULT_PRICE_PREFIX = "price"


def read_api_keys(export_dir=_DEFAULT_EXPORT_DIR, api_key_file=_DEFAULT_API_KEY_FILE):
    filepath = os.path.join(export_dir, api_key_file)
    try:
        with open(filepath, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            api_keys = [(microquant.Source[source], key) for source, key in reader]
            return dict(api_keys)
    except FileNotFoundError:
        return {}


def read_portfolio(
    export_dir=_DEFAULT_EXPORT_DIR, portfolio_file=_DEFAULT_PORTFOLIO_FILE
):
    filepath = os.path.join(export_dir, portfolio_file)
    with open(filepath, "r") as f:
        return [ticker.strip() for ticker in sorted(f) if ticker]


def refresh_news_files(tickers, api_keys, export_dir=_DEFAULT_EXPORT_DIR):
    """load news files, refreshing them from all sources."""
    news = microquant.get_news(tickers, api_keys=api_keys)

    os.makedirs(export_dir, exist_ok=True)

    for ticker, articles in news.items():
        filepath = os.path.join(export_dir, f"{_DEFAULT_NEWS_PREFIX}_{ticker}.csv")

        try:
            read_articles = lambda: pd.read_csv(
                filepath,
                dtype={"time": "str", "title": "str"},
                index_col="time",
                parse_dates=["time"],
            )
            articles = pd.concat([articles, read_articles()])
            articles = news[ticker] = articles.drop_duplicates(subset=["title"])
            os.remove(filepath)
        except FileNotFoundError:
            pass

        articles.to_csv(filepath)

    return news


def refresh_price_files(tickers, api_keys, export_dir=_DEFAULT_EXPORT_DIR):
    del api_keys  # presently unused, will use eventually

    now = datetime.datetime.now()
    now = now.astimezone(ZoneInfo("US/Eastern"))

    timedelta = datetime.timedelta(days=1)

    mcal_nyse = pandas_market_calendars.get_calendar("NYSE")
    schedule = mcal_nyse.schedule(
        (now - timedelta).strftime("%Y-%m-%d"), (now + timedelta).strftime("%Y-%m-%d")
    )

    now = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        market_open = mcal_nyse.open_at_time(schedule, now)
    except ValueError:
        market_open = False

    prices = {}
    for ticker in tickers:
        filepath = os.path.join(export_dir, f"{_DEFAULT_PRICE_PREFIX}_{ticker}.csv")
        logging.info("fetching price for %s, updating %s", ticker, filepath)
        ticker = yf.Ticker(ticker)
        data = [(ticker.fast_info["last_price"], now, market_open)]
        df = pd.DataFrame(data, columns=["price", "time", "market_open"])
        try:
            read_prices = lambda: pd.read_csv(
                filepath, dtype={"price": "float", "time": "str", "market_open": "bool"}
            )
            df = pd.concat([read_prices(), df], ignore_index=True)
        except FileNotFoundError:
            pass
        df.to_csv(filepath, index=False)
        prices[ticker] = df

    return prices


def load_news_files(tickers, export_dir=_DEFAULT_EXPORT_DIR):
    """load news files without refreshing them."""
    news = {}

    for ticker in tickers:
        filepath = os.path.join(export_dir, f"{_DEFAULT_NEWS_PREFIX}_{ticker}.csv")

        try:
            read_articles = lambda: pd.read_csv(
                filepath,
                dtype={"time": "str", "title": "str"},
                index_col="time",
                parse_dates=["time"],
            )

            news[ticker] = read_articles()
        except FileNotFoundError:
            pass

    return news
