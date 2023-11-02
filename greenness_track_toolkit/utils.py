from urllib3 import PoolManager

pool = PoolManager(num_pools=10)


def get_logger():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger("greenness_track_toolkit")


def DEFAULT_START_TIME():
    import datetime
    return datetime.datetime.fromtimestamp(0)


def DEFAULT_END_TIME():
    import datetime
    return datetime.datetime.now()


def to_str_time_args(date):
    import datetime
    return datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M")


def to_str_time_args_to_index(date):
    import datetime
    return datetime.datetime.strftime(date, "%Y-%m-%dT%H:%M")


def to_str_time(date):
    import datetime
    return datetime.datetime.strftime(date, "%Y%m%d%H%M%S.%f")


def format_to_time(str_date: str):
    import datetime
    return datetime.datetime.strptime(str(str_date), "%Y%m%d%H%M%S.%f")


def good_view_time_format(str_date):
    if str_date is None:
        return "-"
    import datetime
    return datetime.datetime.strftime(format_to_time(str_date), "%Y-%m-%d %H:%M:%S")


def time_diff(time1, time2):
    import datetime
    if time2 is None:
        time2 = to_str_time(datetime.datetime.now())
    return (format_to_time(time2) - format_to_time(time1)).seconds