import csv
import microquant
import os
import pandas as pd

_DEFAULT_API_KEY_FILE = "keystore.csv"
_DEFAULT_EXPORT_DIR = "data/"
_DEFAULT_NEWS_PREFIX = "data"
_DEFAULT_PORTFOLIO_FILE = "portfolio.txt"


def read_api_keys(export_dir=_DEFAULT_EXPORT_DIR, api_key_file=_DEFAULT_API_KEY_FILE):
    filepath = os.path.join(export_dir, api_key_file)
    with open(filepath, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        api_keys = [(microquant.Source[source], key) for source, key in reader]
        return dict(api_keys)


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
