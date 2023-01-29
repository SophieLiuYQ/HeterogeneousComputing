import actions
import data_processing

import datetime
import dateutil

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

_PREDICT = flags.DEFINE_bool("p", False, "make predictions")
_REFRESH_NEWS = flags.DEFINE_bool("a", False, "refresh articles")
_REFRESH_PRICES = flags.DEFINE_bool("s", False, "refresh prices")
_UPDATE_WEIGHTS = flags.DEFINE_bool("u", False, "update word weights")
_DAY = flags.DEFINE_string("d", None, "target date")
_TIME_NUM = flags.DEFINE_integer("t", -1, "target time range")
_OPT = flags.DEFINE_string("o", "opt1", "the weighting system to use")
_PRINT_WEIGHTS = flags.DEFINE_bool("z", False, "analyze weights")
# _PRINT_ACCURACY = flags.DEFINE_bool("validate", False, "analyze accuracy")
_USE_GPU = flags.DEFINE_bool("g", False, "use gpu for predictions")


def analyze_weights(use_gpu, weights):
    if weights is None:
        analysis = None
    else:
        if use_gpu:
            analysis = actions.analyze_weights_gpu(weights)
            analysis = analysis if analysis else actions.analyze_weights(weights)
        else:
            analysis = actions.analyze_weights(weights)

    if analysis:
        actions.print_weight_analysis(analysis)
    else:
        raise RuntimeError("could not analyze weights")


def verify_date(date):
    today = datetime.date.today()
    date_parts = date.split("-")
    print("test_date is {}".format(date))
    if len(date_parts) < 3:
        return False

    try:
        test_date = datetime.date(
            int(date_parts[2]), int(date_parts[0]), int(date_parts[1])
        )
    except:
        return False
    if test_date > today:
        return False

    return True


def update_weights(weight_opt, use_gpu, specified_day, time_num, news, prices):
    # Load non-day specific values
    logging.info("Loading non-day specific data before updating word weights")

    # Prepare the days to update word weights for
    days = []

    # First check to see if a day is specified, if it is, set it to the day
    # - If not, use the current day as the specified day
    if not verify_date(specified_day):
        raise RuntimeError("the input day is incorrect", specified_day)

    day = str(dateutil.parser.parse(specified_day)).split()[0]
    days.append(day)

    weights = actions.load_all_word_weights(weight_opt, use_gpu)
    time_nums = range(4) if time_num == -1 else [time_num]

    stock_prices = actions.select_prices(prices)
    logging.info("stock_prices=\n%s", stock_prices)

    for each in days:
        for tn in time_nums:
            logging.info(
                "Updating word weights for %s, across timespan %s-%s",
                each,
                tn * 2 + 8,
                tn * 2 + 10,
            )

            stock_data = actions.select_articles(each, tn, news)
            logging.info("stock_data=\n%s", stock_data)

            actions.update_all_word_weights(
                weight_opt, each, tn, weights, stock_data, stock_prices
            )

    # Save data
    logging.info("Saving word weights")
    actions.save_all_word_weights(weight_opt, weights)
    return weights


def main(argv):
    del argv

    api_keys = data_processing.read_api_keys()
    tickers = data_processing.read_portfolio()

    if _REFRESH_NEWS.value:
        news = data_processing.refresh_news_files(tickers, api_keys)
    elif _UPDATE_WEIGHTS.value:
        news = data_processing.load_news_files(tickers)

    if _REFRESH_PRICES.value:
        prices = data_processing.refresh_price_files(tickers, api_keys)
    elif _UPDATE_WEIGHTS.value:
        prices = data_processing.load_price_files(tickers)

    if _UPDATE_WEIGHTS.value:
        weights = update_weights(
            _OPT.value, _USE_GPU.value, _DAY.value, _TIME_NUM.value, news, prices
        )
    elif _PRINT_WEIGHTS.value:
        weights = actions.load_all_word_weights(_OPT.value, _USE_GPU.value)

    if _PRINT_WEIGHTS.value:
        analyze_weights(_USE_GPU.value, weights)


if __name__ == "__main__":
    app.run(main)
