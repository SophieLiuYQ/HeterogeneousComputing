import datetime
import math
import re
import struct
import zoneinfo

from absl import logging

MAX_WORDS_PER_LETTER = 2500
TIMEZONE = zoneinfo.ZoneInfo("US/Central")


def load_all_word_weights(weight_opt, use_gpu):
    logging.info("Loading word weights for weighting option: %s", weight_opt)

    total_up = 0
    total_words_up = 0
    total_down = 0
    total_words_down = 0
    words_by_letter = []
    num_words_by_letter = []

    # For each letter, add a MAX_WORDS_PER_LETTER word array of 28 characters each with the last 12 characters for a float (weight of the word) and int (number of occurences)
    # - abcdefghijklmnopq0.32########
    for _ in range(0, 26):
        letter_words = bytearray(28 * MAX_WORDS_PER_LETTER)
        words_by_letter.append(letter_words)
        num_words_by_letter.append(0)

    # Open the file to be loaded
    try:
        file = open("./data/word_weight_data_" + weight_opt + ".txt", "r+")
    except IOError as error:
        logging.warning("Could not load word weights, Error: " + str(error))
        return (
            total_up,
            total_words_up,
            total_down,
            total_words_down,
            words_by_letter,
            num_words_by_letter,
        )

    # Iterate over all the lines in the file
    letter_index = 0
    first = 0
    for lines in file:
        # If its the first line and option 2, load the total words up or down
        if first == 0 and weight_opt == "opt2":
            temp = lines.split()
            total_up = int(temp[0])
            total_down = int(temp[1])
            first = 1
        elif first == 1 and weight_opt == "opt2":
            temp = lines.split()
            total_words_up = int(temp[0])
            total_words_down = int(temp[1])
            first = 2

        # If the first character is not a '-', it indicates a letter change
        if lines[:1] != "-":
            letter_index = ord(lines[:1]) - 97
            logging.debug("Loading words beginning with: " + lines[:1])

        # If not a '-', pack up the data, store it, and update the number of words
        else:
            # Get the data
            data = lines.split()
            logging.debug("loaded data: %s", data)

            # Store the data
            if use_gpu:
                struct.pack_into(
                    "16s f i i",
                    words_by_letter[letter_index],
                    num_words_by_letter[letter_index] * 28,
                    data[1].encode("ascii").decode("ascii", "ignore").encode("utf-8"),
                    float(data[2]),
                    int(data[3]),
                    int(data[4]),
                )
            else:
                struct.pack_into(
                    "16s f i i",
                    words_by_letter[letter_index],
                    num_words_by_letter[letter_index] * 28,
                    data[1].encode("utf-8"),
                    float(data[2]),
                    int(data[3]),
                    int(data[4]),
                )
            num_words_by_letter[letter_index] += 1

    file.close()

    return (
        total_up,
        total_words_up,
        total_down,
        total_words_down,
        words_by_letter,
        num_words_by_letter,
    )


def analyze_weights_gpu(weights):
    del weights
    return None


def analyze_weights(weights):
    logging.info("Analyzing weights for distribution")

    weights_all = []
    weights_all_o = []
    weight_average = 0
    weight_stdev = 0
    weight_sum = 0
    weight_max = 0
    weight_min = 1
    weight_count = 0
    weight_average_o = 0
    weight_stdev_o = 0
    weight_sum_o = 0
    weight_count_o = 0

    _, _, _, _, words_by_letter, num_words_by_letter = weights

    # Iterate through each letter
    for letter in range(0, 26):
        logging.debug("- Analyzing word weights for letter: " + chr(letter + 97))

        # Iterate through each word for that letter
        for elements in range(0, num_words_by_letter[letter]):
            # For each word, unpack the word from the buffer
            raw_data = struct.unpack_from(
                "16s f i i", words_by_letter[letter], elements * 28
            )
            weight = float(raw_data[3]) / raw_data[2]

            # Add it to the list of weights to be analyzed later (for standard deviation)
            weights_all.append(weight)
            for n in range(0, raw_data[2]):
                weights_all_o.append(weight)

            # Update sum, max, min, and count
            weight_count += 1
            weight_sum += weight

            weight_count_o += raw_data[2]
            weight_sum_o += weight * raw_data[2]

            if weight > weight_max:
                weight_max = weight

            if weight < weight_min:
                weight_min = weight

    # If no words have been found with weights return false
    if weight_count == 0:
        logging.error("Could not find any words with weights to analyze")
        return None

    # Once all weights have been iterated through, calculate the average
    weight_average = weight_sum / weight_count
    weight_average_o = weight_sum_o / weight_count_o

    # Calculate the standard deviation
    running_sum = 0
    for weights in weights_all:
        running_sum += (weights - weight_average) * (weights - weight_average)

    running_sum_o = 0
    for weights in weights_all_o:
        running_sum_o += (weights - weight_average_o) * (weights - weight_average_o)

    weight_stdev = math.sqrt(running_sum / (weight_count - 1))
    weight_stdev_o = math.sqrt(running_sum_o / (weight_count_o - 1))

    return (
        weight_average,
        weight_stdev,
        weight_sum,
        weight_count,
        weight_max,
        weight_min,
        weight_average_o,
        weight_stdev_o,
        weight_sum_o,
        weight_count_o,
    )


def print_weight_analysis(analysis):
    (
        weight_average,
        weight_stdev,
        weight_sum,
        weight_count,
        weight_max,
        weight_min,
        weight_average_o,
        weight_stdev_o,
        weight_sum_o,
        weight_count_o,
    ) = analysis

    logging.info("- Analysis finished with:")
    logging.info("-- avg: " + str(weight_average))
    logging.info("-- std: " + str(weight_stdev))
    logging.info("-- sum: " + str(weight_sum))
    logging.info("-- cnt: " + str(weight_count))
    logging.info("-- max: " + str(weight_max))
    logging.info("-- min: " + str(weight_min))
    logging.info("-- avg_o: " + str(weight_average_o))
    logging.info("-- std_o: " + str(weight_stdev_o))
    logging.info("-- sum_o: " + str(weight_sum_o))
    logging.info("-- cnt_o: " + str(weight_count_o))


def update_word(ticker, option, word_upper, day, time_num, weights, stock_prices):
    # Make the word lowercase and get the length of the word
    word = word_upper.lower()
    word_len = len(word) if len(word) < 16 else 16

    # Find the letter index for the words array
    index = ord(word[:1]) - 97
    if index < 0 or index > 25:
        logging.warning("-- Could not find the following word in the database: " + word)
        return

    (
        _,
        total_words_up,
        _,
        total_words_down,
        words_by_letter,
        num_words_by_letter,
    ) = weights

    # Get the array containing words of the right letter
    letter_words = words_by_letter[index]
    num_letter_words = num_words_by_letter[index]

    # Search that array for the current word
    found = False
    for ii in range(0, num_letter_words):
        # Get the current word data to be compared
        test_data = struct.unpack_from("16s f i i", letter_words, ii * 28)
        temp_word = test_data[0].decode("utf_8").split("\0", 1)[0]

        # Check if the word is the same
        if len(temp_word) == word_len and temp_word == word[:word_len]:
            # If it is the same, mark it as found and update its values
            # weight = (weight * value + increase)/(value + increase)
            found = True

            if not ticker in stock_prices:
                logging.error(
                    "-- Could not find stock " + ticker + " in stock price data"
                )
                return
            if not day in stock_prices[ticker]:
                logging.error(
                    "-- Could not find stock price data for " + ticker + " on " + day
                )
                return

            change = (
                stock_prices[ticker][day][time_num][1]
                - stock_prices[ticker][day][time_num][0]
            )

            # Option 1: 1 for up, 0 for down, average the ups and downs for weights
            # weight = num_up / total
            # extra1 = total
            # extra2 => unused
            if option == "opt1":
                if change > 0:
                    weight = test_data[1]
                    extra1 = test_data[2] + 1
                    extra2 = test_data[3] + 2

                else:
                    weight = test_data[1]
                    extra1 = test_data[2] + 1
                    extra2 = test_data[3]

            # Option 2: Bayesian classifier, probability of word given a label
            # weight => Unused, calculated seperately
            # extra1 = num_up
            # extra2 = num_down
            elif option == "opt2":
                if change > 0:
                    weight = test_data[1]
                    extra1 = test_data[2] + 1
                    extra2 = test_data[3]
                    total_words_up += 1
                else:
                    weight = test_data[1]
                    extra1 = test_data[2]
                    extra2 = test_data[3] + 1
                    total_words_down += 1

            struct.pack_into(
                "16s f i i",
                letter_words,
                ii * 28,
                word.encode("utf-8"),
                weight,
                extra1,
                extra2,
            )

            logging.debug(
                "-- Updated "
                + word
                + " using "
                + option
                + " with weight of "
                + str(weight)
                + " and occurences of "
                + str(extra1)
                + ", "
                + str(extra2)
            )
            break

    if not found:
        # Get whether the stock went up or down
        # weight is automatically 1 or 0 for the first one
        change = (
            stock_prices[ticker][day][time_num][1]
            - stock_prices[ticker][day][time_num][0]
        )

        if option == "opt1":
            if change > 0:
                weight = 0
                extra1 = 1
                extra2 = 2
            else:
                weight = 0
                extra1 = 1
                extra2 = 0

        elif option == "opt2":
            if change > 0:
                weight = 0
                extra1 = 1
                extra2 = 0
                total_words_up += 1
            else:
                weight = 0
                extra1 = 0
                extra2 = 1
                total_words_down += 1

        # Pack the data into the array
        struct.pack_into(
            "16s f i i",
            letter_words,
            num_letter_words * 28,
            word.encode("utf-8"),
            weight,
            extra1,
            extra2,
        )
        num_words_by_letter[index] += 1
        logging.debug(
            "-- Added "
            + word
            + " with weight of "
            + str(weight)
            + " and occurences of "
            + str(extra1)
            + ", "
            + str(extra2)
        )


def update_all_word_weights(option, day, time_num, weights, stock_data, stock_prices):
    """
    | |
    |_|
    |_|_______________
    |3|_0_|__1__|__2__
    |_|         |'hey'|
    | |         | 0.8  |
    | |         | 80  |

    """
    if time_num == 0:
        time_span = ("8:30", "10:00")
    elif time_num == 1:
        time_span = ("10:00", "12:00")
    elif time_num == 2:
        time_span = ("12:00", "14:00")
    else:
        time_span = ("14:00", "15:30")

    logging.info("Updating word weights for {} ~ {}".format(time_span[0], time_span[1]))

    _, _, _, _, words_by_letter, num_words_by_letter = weights

    # If the weighting arrays are empty, create them
    if len(words_by_letter) == 0 or words_by_letter == 0:
        logging.debug("- Could not find data structure for word weights so creating it")

        # For each letter, add a MAX_WORDS_PER_LETTER word array of 28 characters each with the last 8 characters for a float (weight of the word) and int (number of occurences)
        # - abcdefghijklmnopq0.32####
        for _ in range(0, 26):
            letter_words = bytearray(28 * MAX_WORDS_PER_LETTER)
            words_by_letter.append(letter_words)
            num_words_by_letter.append(0)

    # At this point, the weighting arrays are initialized or loaded
    # - Next step is to iterate through articles and get the words
    for ticker, articles in stock_data.items():
        logging.debug("- Updating word weights for: " + ticker)

        if not ticker in stock_data:
            logging.warning("- Could not find articles loaded for " + ticker)
            continue

        cpu = 0

        # For each stock, iterate through the articles
        for article in articles:
            # Get the text (ignore link)
            text = article

            # Get an array of words with two or more characters for the text
            words_in_text = re.compile("[A-Za-z][A-Za-z][A-Za-z]+").findall(text)

            # Update each word
            for each_word in words_in_text:
                update_word(
                    ticker, option, each_word, day, time_num, weights, stock_prices
                )

    # Add all the weights to the cpu weight array
    update_outputs = []
    for letter in range(0, 26):
        letter_words = words_by_letter[letter]

        for each_word in range(0, num_words_by_letter[letter]):
            test_data = struct.unpack_from("16s f i i", letter_words, each_word * 28)

            update_outputs.append(test_data[2])
            update_outputs.append(test_data[3])

    return update_outputs


def select_prices(prices):
    stock_prices = {}

    def store_price(ticker, date, idx, price):
        ticker_dict = stock_prices.get(ticker, {})
        date_dict = ticker_dict.get(date, {})
        idx_list = date_dict.get(idx, [])

        idx_list.append(price)

        date_dict[idx] = idx_list
        ticker_dict[date] = date_dict
        stock_prices[ticker] = ticker_dict

    for ticker, df in prices.items():
        # create stock_prices dic for current stock
        cur_day, price_prev = "", 0
        for row in df.itertuples():
            _, price, dt, market_open = row
            if not market_open:
                continue
            # get price and datetime, e.g., price = 180.17, date_time = 12/13/21
            price = round(price, 2)
            dt = dt.replace(tzinfo=TIMEZONE)
            # reformat time, e.g., date = 12/13/21, localtime = 15:15
            date = dt.strftime("%Y-%m-%d")
            localtime = dt.strftime("%H:%M")
            # formulate date
            if date != cur_day:
                if price_prev != 0:
                    store_price(ticker, cur_day, 3, price_prev)
                cur_day = date
                cur_hour = 8
                store_price(ticker, date, 0, price)
            # formulate price to be 2 hr interval
            hour = int(localtime.split(":")[0])
            if hour != cur_hour and hour != (cur_hour + 1):
                store_price(ticker, date, (cur_hour - 8) // 2, price)
                cur_hour = hour
                store_price(ticker, date, (cur_hour - 8) // 2, price)
            price_prev = price
        store_price(ticker, date, (cur_hour - 8) // 2, price)

    return stock_prices


def select_articles(day, time_num, news):
    def as_datetime(day, hour, minute="00"):
        dt = datetime.datetime.strptime(
            f"{day} {hour}:{minute}", "%Y-%m-%d %H:%M"
        ).replace(tzinfo=TIMEZONE)
        return dt

    if time_num == 0:
        # at market open, include everything from yesterday after 15:30 CT
        start_time = as_datetime(day, 15, 30) - datetime.timedelta(days=1)
    else:
        start_time = as_datetime(day, time_num * 2 + 6)

    end_time = as_datetime(day, time_num * 2 + 8)

    logging.info("using articles from {%s}-{%s}", start_time, end_time)

    stock_data = {}

    for ticker, articles in news.items():
        stock_data[ticker] = []

        for row in articles.itertuples():
            date_time, title = row
            date_time = date_time.to_pydatetime()

            if date_time >= start_time and date_time <= end_time:
                stock_data[ticker].append(title)
            elif date_time < start_time:
                break  # terminate search early (ordered data)

    return stock_data


def save_all_word_weights(option, weights):
    logging.info("Saving word weights for weighting option " + option)

    (
        total_up,
        total_words_up,
        total_down,
        total_words_down,
        words_by_letter,
        num_words_by_letter,
    ) = weights

    file = open("./data/word_weight_data_" + option + ".txt", "w")

    # First write the global data to the file
    if option == "opt2":
        file.write(str(total_up) + " " + str(total_down) + "\n")
        file.write(str(total_words_up) + " " + str(total_words_down) + "\n")

    # Iterate through all letters and words in each letter and write them to a file
    for first_letter in range(0, 26):
        # Write the letter to the file
        file.write(chr(first_letter + 97) + "\n")

        logging.debug(
            "- Saving word weights for words starting with: " + chr(first_letter + 97)
        )

        # For each letter, iterate through the words saved for that letter
        for words in range(0, num_words_by_letter[first_letter]):
            # For each word, unpack the word from the buffer
            raw_data = struct.unpack_from(
                "16s f i i", words_by_letter[first_letter], words * 28
            )
            temp_word = raw_data[0].decode("utf-8")

            # Write the data to the file
            file.write(
                "- "
                + temp_word.split("\0", 1)[0]
                + " "
                + str(raw_data[1])
                + " "
                + str(raw_data[2])
                + " "
                + str(raw_data[3])
                + "\n"
            )

    file.close()
