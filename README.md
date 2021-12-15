Dec 6 - Sophie's task: rewrite the cpu part so that the "headers" are pulled and analyzed, not the entire article.
Dec 7 - Sophie's task: grab and compile the news and prices for the past months. Write the CPU listener to collect data every hr, starting from Dec 8.
Dec 8 - Sophie's task: run and test the CPU listener, pushed to remote server, and updated the data in git Data Branch from 9am to 5pm EST (8am to 4pm CT). The price is updated every 15 mins, the news is updated every hr. 
Dec 9 - Sophie's task: no task scheduled for the day. Doing other HWs.
Dec 10 - Sophie's task: look into the algorithm and try to run with our data. 
Dec 13 - Sophie's tasks: check load_stock_price and load_article and see whether can run it
Dec 14 -Sophie's tasks: attemped to rewrite and clean up the code but failed. Instead, set up a gcloud auto runner to predict and update weights from Dec15 to Dec17. 

Dec  8~9 - Xiao's task: search statistic prediction algorithm; reduce the algorithm to three; modify load_stock_price according to the csv data format.
Dec 10 - Xiao's task: modify (use_for_test.py) the way to load word weights according to our project.
Dec 13 - Xiao's task: modify (use_for_test.py) load_stock_price and load_article according to the file format.
         divide the trading hours into 0: 8-10; 1: 10-12; 2: 12-14; 3: 14-16
         (actually for time > 14, I classify it as type 3, as it is used for predicting tomorrow 8-10)
         There is something inconsistent here, need to fix!!!
