'''
Verifies that a date is valid and in the right format
'''
import datetime

def verify_date(date):
    today = datetime.date.today()
    date_parts = date.split('-')

    if len(date_parts) < 3:
        return False

    try:
        test_date = datetime.date(int(date_parts[2]), int(date_parts[0]), int(date_parts[1]))
    except:
        return False
    # print("test_date is {}".format(test_date))
    if test_date > today:
        return False

    return test_date
