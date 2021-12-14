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
    # directory is './data/data_{}'
    for ticker in STOCK_TAGS:
        # start = time.time()

        # Try to open the file for that stock from the given directory
        logging.debug('Starting to load articles for: ' + ticker)
        try:
            file = open('./data/data_{}.csv'.format(ticker), 'r')
        except IOError as error:
            logging.warning('Could not load articles for: ' + ticker + ', Error: ' + str(error))
            continue

        # skip the first line as it is the header
        file.readline()
        # Prepare variables to save the article data
        stock_data[ticker] = []
        need_time = (time_num*2 + 8, time_num*2 + 10)
        for line in file:
            data = line.strip().rsplit(",", 1)
            # data[0] is the news header and data[1] is the datetime
            date_time = data[1]
            date_time = date_time.split()
            if len(date_time) == 0:
                continue
            date = (str(parse(date_time[0])).split())[0]
            localtime = "00:00" if (len(date_time) == 1) else date_time[1]
            hour = int(localtime.split(":")[0])
            # print(hour)
            if date == day:
                print(data[0], hour, need_time)
                if time_num != 3:
                    if hour >= need_time[0] and hour < need_time[1]:
                        stock_data[ticker].append(data[0])
                else:
                    # if we load articles for 14-16,we actually load all the aricles after 14 pm
                    # , because the data is used for tomorrow morning
                    if hour >= need_time[0]:
                        stock_data[ticker].append(data[0])
        # print(current_article)
        print(stock_data[ticker])
        # if len(stock_data[ticker]) < SUCCESS_THREASHOLD:
        #     logging.error('Could not load the threshold (' + str(
        #         SUCCESS_THREASHOLD) + ') number of articles. Either not saved or another error occured.')
        #     return False
    print(stock_data)
    return True
