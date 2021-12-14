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


def load_articles(day, time_num, STOCK_TAGS):
    stock_data = {}
    # directory is './data/data_{}'
    for ticker in STOCK_TAGS:

        # Try to open the file for that stock from the given directory
        logging.debug("Starting to load articles for: " + ticker)
        try:
            file = open("./data/data_{}.csv".format(ticker), "r")
        except IOError as error:
            logging.warning(
                "Could not load articles for: " + ticker + ", Error: " + str(error)
            )
            continue

        # skip the first line as it is the header
        file.readline()
        # Prepare variables to save the article data
        stock_data[ticker] = []

        need_time = (time_num * 2 + 8, time_num * 2 + 10)
        for line in file:
            data = line.strip().rsplit(",", 1)

            # data[0] is the news header and data[1] is the datetime
            date_time = data[1]
            # parse the second element to only include the date part, i.e., date = 12/13/21
            date = str(parse(date_time.split()[0])).split()[0]
            # parse the second element to only include the time part, i.e., localtime = 15:15
            localtime = date_time.split()[1]
            # parse the localtime to only include the hrs, i.e., hour = 15
            hour = int(localtime.split(":")[0])

            # if we load articles for 14-16,we actually load all the aricles after 14 pm
            # , because the data is used for tomorrow morning
            if date == day:
                if time_num != 3:
                    if hour >= need_time[0] and hour < need_time[1]:
                        stock_data[ticker].append(data[0])
                else:
                    if hour >= need_time[0]:
                        stock_data[ticker].append(data[0])
        # print(current header)
        file.close()
    print(stock_data)
    return stock_data
