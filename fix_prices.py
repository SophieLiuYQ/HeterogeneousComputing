from datetime import date, datetime
import glob
import pandas as pd
import datetime

delta = datetime.timedelta(hours=1)

prefix="data_"

# find all csv file in current directory
for file in glob.glob("{0}*.csv".format(prefix)):
    ## extract ticker from file name
    ticker=file[len(prefix):file.rfind(".")]
    print("info> reading", ticker, "from", file)
    df = pd.read_csv(file)
    ## find row where time = desire time
    for i in range(len(df)):
        t = df.loc[i,'time']
    ##df.drop(df.index[df['time'] == '12/09/21 08:01'], inplace=True)
    ##df.drop(df.index[df['time'] == '12/09/21 08:16'], inplace=True)
    ##df.drop(df.index[df['time'] == '12/09/21 15:15'], inplace=True)
    ##df.drop(df.index[df['time'] == '12/09/21 15:31'], inplace=True)
    ##df.drop(df.index[df['time'] == '12/09/21 15:45'], inplace=True)
    ##df.drop(df.index[df['time'] == '12/09/21 16:01'], inplace=True)
    ##df.drop(df.index[df['time'] == '12/09/21 16:16'], inplace=True)
    ##df.drop(df.index[df['time'] == '12/09/21 16:31'], inplace=True)
    ##df.drop(df.index[df['time'] == '12/09/21 16:46'], inplace=True)
        #if in AM/PM format
        if 'AM' in t or 'PM' in t:
          #   read the date
            dt = datetime.datetime.strptime(t, '%Y-%m-%d %I:%M:%S %p')
            # and update it to 24-hr format
            df.loc[i,'time'] = dt.strftime('%m/%d/%y %H:%M')
    df.to_csv(file, index=False)