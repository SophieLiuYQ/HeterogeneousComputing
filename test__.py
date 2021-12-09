import logging
STOCK_TAGS = ['FB',
              'MSFT',
              'AAPL',
              'NVDA',
              'AMD',
              'XLNX',
              'QCOM',
              'MU']

stock_prices = {}
total_down = 0
total_up = 0

for stock in STOCK_TAGS:
    try:
        file = open("./past_data/{}.csv".format(stock), 'r')
    except IOError as error:
        logging.warning('- Could not load stock prices for {}, Error: '.format(stock) + str(error))

    # skip the first line as it is the header
    file.readline()
    stock_prices[stock] = {}
    for line in file:
        data = line.strip().split(",")
        open_price = round(float(data[1]), 5)
        close_price = round(float(data[4]), 5)
        stock_prices[stock][data[0]] = (open_price, close_price)

        if open_price > close_price:
            total_down += 1
        else:
            total_up += 1
    file.close()