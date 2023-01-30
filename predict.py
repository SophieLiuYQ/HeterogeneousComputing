"""if this script is run within 15 minutes of a
   new time range, try to predict its outcome"""

import datetime
import os
import subprocess
import sys
import zoneinfo

from absl import app
from absl import flags

FLAGS = flags.FLAGS

_DATE_FORMAT = "%m-%d-%y"
_TIME_FORMAT = "%m/%d/%y, %H:%M:%S %z"
_TIME_FLAG = flags.DEFINE_string(
    "t", None, f"predict for specified time (format: '{_TIME_FORMAT}')"
)

_START_TIMES = [
    datetime.time(8, 30),
    datetime.time(10, 0),
    datetime.time(12, 0),
    datetime.time(14, 0),
]
_START_TIMES_TIMEZONE = zoneinfo.ZoneInfo("US/Central")


def get_timedelta(dt, t):
    t = dt.replace(hour=t.hour, minute=t.minute)
    td = (dt - t) if (dt >= t) else (t - dt)
    return td


def get_timenum(dt, tol_minutes=15):
    dt = dt.astimezone(_START_TIMES_TIMEZONE)

    for i, t in enumerate(_START_TIMES):
        td = get_timedelta(dt, t).seconds / 60

        if td < tol_minutes:
            return i

    return None


def run_cmd(args):
    print("$", *args)
    args = [str(x) for x in args]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    outs, _ = proc.communicate()
    outs = outs.decode("utf-8")
    if proc.returncode:
        print(outs, file=sys.stderr)
        sys.exit(proc.returncode)
    return outs


def main(argv):
    del argv

    os.makedirs("output", exist_ok=True)

    if _TIME_FLAG.value is None:
        dt = datetime.datetime.now()
    else:
        dt = datetime.datetime.strptime(_TIME_FLAG.value, _TIME_FORMAT)

    timenum = get_timenum(dt)
    if timenum is None:
        sys.exit(0)

    date = dt.strftime(_DATE_FORMAT)
    predict_args = [
        "python3",
        "use_for_test_better_result.py",
        "-d",
        date,
        "-t",
        timenum,
        "-p",
    ]

    prev_timenum = timenum - 1 if timenum > 0 else 3
    prev_date = (
        date
        if prev_timenum < 3
        else (dt - datetime.timedelta(days=1)).strftime(_DATE_FORMAT)
    )

    update_args = [
        "python3",
        "use_for_test_better_result.py",
        "-d",
        prev_date,
        "-t",
        prev_timenum,
        "-u",
    ]

    print(run_cmd(update_args))
    print("")
    print(run_cmd(predict_args))


if __name__ == "__main__":
    app.run(main)
