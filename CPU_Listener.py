from bs4 import BeautifulSoup as BS
import requests
import datetime
import os 

## getHeading returns heading (x.getText()) and time (spans[0])
def getHeading(x):
    now = datetime.datetime.now()
    return (x.getText(), now.strftime('%m-%d-%y %I:%M %p'))

def PullArticle(tickers):

	print('Pulling articles')

	# Iterate through the stock tags
	tickers = ['FB','MSFT','AAPL','NVDA','AMD','XLNX','QCOM','MU']
	for ticker in tickers:
		file = open(ticker + '.txt', 'w')

		## Make request to google, parse the first and the second page of the news via BeautifulSoup
		r1 = requests.get('https://www.google.com/search?q=' + ticker + '&rlz=1C5CHFA_enUS906US906&source=lnms&tbm=nws&sa=X&ved=2ahUKEwjtx66qpdD0AhUrk4kEHTxNBHwQ_AUoAnoECAIQBA&biw=1440&bih=701&dpr=2', timeout=2)
		r2 = requests.get('https://www.google.com/search?q=' + ticker + '&rlz=1C5CHFA_enUS906US906&tbm=nws&ei=TriuYfmAPeKYptQP1ZicuAY&start=10&sa=N&ved=2ahUKEwj5382exND0AhVijIkEHVUMB2cQ8tMDegQIARA3&biw=1440&bih=701&dpr=2', timeout = 2)
		data1 = BS(r1.text, 'html.parser')
		data2 = BS(r2.text, 'html.parser')

		## combine headings and times from the first and the second page of the news
		data_processed = [ getHeading(x) for x in data1.find_all('h3')] + [ getHeading(x) for x in data2.find_all('h3') ]

		## write everything into a file
		for element in data_processed:
			file.write(' '.join(str(s) for s in element) + '\n')
		file.close()

#print(datetime.datetime.now())
	return file

def CPU_listener():

    tickers = ['FB','MSFT','AAPL','NVDA','AMD','XLNX','QCOM','MU']
    for ticker in tickers:
        file = open(ticker + '.txt', 'r+')
        for lines in file:
            

        