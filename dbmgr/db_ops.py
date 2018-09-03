import itertools
from utils import write_and_restart_line

from .models import SANDP500, DJI, USTwitterNewsFeed
from .data_config import INDICES, DICTIONARY
from pandas import read_csv


RAW_STOCK_FEATURE_CSV = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
RAW_STOCK_FEATURE_DB = ['date', 'open', 'high', 'low', 'close', 'volume']


def get_csvlocation_and_dbmodel_from_stock_code(stock_code):

    if stock_code not in INDICES:
        assert False, "unsupported stock code entered"

    csv_location = INDICES[stock_code]['loc']
    dbmodel = INDICES[stock_code]['dbmodel']

    return csv_location, dbmodel


def fetch_and_save_stock_code_to_db(stock_code):

    csv_location, dbmodel = get_csvlocation_and_dbmodel_from_stock_code(stock_code)

    spinner = itertools.cycle(['|', '/', '-', '\\'])
    print("Please wait while it is loading... ", end="")
    for df in read_csv(csv_location, names=RAW_STOCK_FEATURE_CSV, chunksize=1, skiprows=1):
        kwargs = dict(zip(RAW_STOCK_FEATURE_DB, df.values[0]))
        try:
            dbmodel.objects.create(**kwargs)
        except:
            pass

        if df.index.get_values()[0] % 50 == 0:
            write_and_restart_line(next(spinner))

    write_and_restart_line(stock_code + " was saved to database")


def init_twitter_keyword_dictionary():
    """ build twitter keyword dictionary from text2 feed """
    dictionary_loc = DICTIONARY['TWITTER']['KEYWORD_LIST']
    keyword_list = set()

    # for feed in USTwitterNewsFeed.objects.all():
    #     if feed is not None:
    #         keyword_list = keyword_list | set((feed.text2).split())
    neg_news = USTwitterNewsFeed.objects.filter(created_datetime__range=["2017-11-13", "2017-11-14"])   # 1280
    pos_news = USTwitterNewsFeed.objects.filter(created_datetime__range=["2017-11-24", "2017-11-27"])   # 328

    for feed in neg_news:
        if feed.text2 is not None:
            keyword_list = keyword_list | set((feed.text2).split())
    for feed in pos_news:
        if feed.text2 is not None:
            keyword_list = keyword_list | set((feed.text2).split())

    with open(dictionary_loc, 'a') as keyword_file:
        for keyword in keyword_list:
            try:
                keyword_file.write(keyword + '\n')
            except:
                pass

    return keyword_list


# TODO: GENERALIZE FUNCTION
def get_serialize_stock_code_test():
    dji = list(DJI.objects.values_list('date', 'open', 'high', 'low', 'close', 'volume'))

    return dji
