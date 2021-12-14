'''
To update word weights for articles and prices for the day specified, 
use (must have already pulled weights and prices):
./stock_market_prediction.py -d mm-dd-yyyy\n')

To update word weights for articles and prices for the day 
specified and time specified(0:8-10,1:10-12,2:12-14,3:14-16), 
use (must have already pulled weights and prices):
./stock_market_prediction.py -d mm-dd-yyyy -t time_num

Currently available weighting options are opt1 and opt2. 
opt1 uses average with 1 for a word seen with up and 0 with a word seen with down.
opt2 uses a Naive Bayes classifier.
'''

from dateutil.parser import parse
import logging

def load_stock_prices(day, time_num, STOCK_TAGS):
    stock_prices = {}
    for ticker in STOCK_TAGS:
        try:
            file = open("./data/price_{}.csv".format(ticker), 'r')
        except IOError as error:
            logging.warning('- Could not load stock prices for {}, Error: '.format(ticker) + str(error))

        # skip the first line as it is the header
        file.readline()
        stock_prices[ticker] = []
        # create stock_prices dic for current stock
        cur_day = ""
        price_prev = 0
        for line in file:
            # parse data on the ","
            data = line.strip().split(",")
            # get price and datetime, e.g., price = 180.17, date_time = 12/13/21
            price = round(float(data[0]), 2)
            date_time = data[1]
            # reformat time, e.g., date = 12/13/21, localtime = 15:15
            date = str(parse(date_time.split()[0])).split()[0]
            localtime = date_time.split()[1]
            # formulate date 
            if date != cur_day:
                if price_prev != 0:
                    stock_prices[ticker][cur_day][3].append(price_prev)
                cur_day = date
                cur_hour = 8
                stock_prices[ticker][date] = {}
                stock_prices[ticker][date][0] = [price]
            # formulate price to be 2 hr interval
            hour = int(localtime.split(":")[0])
            if hour != cur_hour and hour != (cur_hour+1):
                stock_prices[ticker][date][(cur_hour-8)//2].append(price)
                cur_hour = hour
                stock_prices[ticker][date][(cur_hour-8)//2] = [price]
            price_prev = price
        file.close()

    return stock_prices