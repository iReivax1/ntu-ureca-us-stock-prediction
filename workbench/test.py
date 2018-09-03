import multiprocessing as mp
import numpy as np
import time
from decorators import log_time_taken


def generate_twitter_feed_influence_score():

    from dbmgr.data_config import DICTIONARY
    from dbmgr.models import USTwitterNewsFeed
    from sentiment.feed_influence import InfluenceCalculator

    dictionary_loc = DICTIONARY['TWITTER']['KEYWORD_LIST']
    with open(dictionary_loc) as f:
        keywords = f.readlines()

    keywords = [x.strip('\n') for x in keywords]
    pos_news = USTwitterNewsFeed.objects.filter(type="pos")     # 1280 news
    neg_news = USTwitterNewsFeed.objects.filter(type="neg")     # 1719 news
    # pos_news = USTwitterNewsFeed.objects.filter(created_datetime__range=["2017-11-24", "2017-11-27"])
    # neg_news = USTwitterNewsFeed.objects.filter(created_datetime__range=["2017-11-13", "2017-11-14"])

    ic = InfluenceCalculator(keywords, pos_news, neg_news)
    ic.calculate_kword_count()
    ic.calculate_posterior_probability()

    for feed in USTwitterNewsFeed.objects.all():
        feed.sentiment = ic.calculate_text_influence(feed.text2)
        feed.save()


def calculate_news_feed_sentiment_per_day():
    from django.db.models import Avg, Max, Min
    from dbmgr.models import USTwitterNewsFeed

    news = USTwitterNewsFeed.objects.filter(created_datetime__range=["2017-11-01", "2017-12-01"])

    nnews = news.extra(select={'day': 'date( created_datetime )'}). \
        values('day').annotate(avg_sentiment=Avg('sentiment')).order_by('day')


def execute_gru2_twitter_sentiment():

    from django.db.models import Avg, Max, Min
    from dbmgr.models import DJI, USTwitterNewsFeed
    import pandas as pd
    import logging
    from nn_models import gru
    from utils import normalize, calculate_error

    # PREPARE PD-FRAME PRICE DATA
    dji = list(DJI.objects.values_list('date', 'open', 'high', 'low', 'close', 'volume'))
    result = np.asarray(dji)
    description = np.asarray(['date', 'open', 'high', 'low', 'close', 'volume'])

    data = pd.DataFrame(data=result, columns=description[:])
    data.set_index('date', inplace=True)
    data.index = data.index.map(str)

    for col in data.columns:
        data[col] = data[col].astype(dtype=np.float64)

    price_change = data.iloc[1:, [3]].values - data.iloc[:-1, [3]].values
    p_change = price_change / data.iloc[:-1, [3]].values
    price_change = np.concatenate(([[0]], price_change))
    p_change = np.concatenate(([[0]], p_change))
    data['price_change'] = price_change
    data['p_change'] = p_change * 100
    data = data.drop(data.index[0])     # remove first row

    # CALCULATE NORMALIZE Y
    train_y = data['close'].values / data['close'].values[0] - 1

    # DETERMINE AMOUNT OF DATA TO TRAIN ON
    RATIO = 1.0
    BGN_LENGTH = np.int(np.ceil(data.shape[0] * RATIO))

    # SELECT FEATURES FOR THE MODEL
    norm_data = pd.DataFrame(index=data.index)
    norm_data['close'] = data['close']
    norm_data['volume'] = data['volume']

    # TRAIN_Y INDEXED BY DATE
    y_data = pd.DataFrame(index=data.index)
    y_data['train_y'] = train_y

    # INCORPORATE SENTIMENT AS FEATURES -- TEST DATA
    news = USTwitterNewsFeed.objects.filter(created_datetime__range=["2018-02-01", "2018-02-28"])
    nnews = news.extra(select={'day': 'date( created_datetime )'}). \
        values('day').annotate(avg_sentiment=Avg('sentiment')).order_by('day')
    for news in nnews:
        norm_data.loc[str(news['day']), 'sentiment'] = news['avg_sentiment']

    # INCORPORATE SENTIMENT AS FEATURES -- TRAINING DATA
    news = USTwitterNewsFeed.objects.filter(created_datetime__range=["2017-11-01", "2017-12-01"])
    nnews = news.extra(select={'day': 'date( created_datetime )'}). \
        values('day').annotate(avg_sentiment=Avg('sentiment')).order_by('day')
    for news in nnews:
        norm_data.loc[str(news['day']), 'sentiment'] = news['avg_sentiment']

    test_data = norm_data["2018-02-26":"2018-02-02"]
    test_data = np.asarray(test_data, dtype=np.float64)
    test_data_y = np.asarray(y_data["2018-02-26":"2018-02-02"]["train_y"])

    norm_data = norm_data["2017-11-30":"2017-11-01"]
    norm_data = np.asarray(norm_data, dtype=np.float64)
    norm_data_y = np.asarray(y_data["2017-11-30":"2017-11-01"]["train_y"])

    # NORMALIZE FEATURES
    test_data = normalize(test_data)
    norm_data = normalize(norm_data)

    # CALCULATE MODEL DIMENSIONS
    input_dim = norm_data.shape[1]
    hidden_dim = input_dim * 2

    # TRAIN MODEL AND EVALUATE TRAINING LOSS ON NOVEMBER
    model = gru.GRU_2(input_dim, hidden_dim, seed=0)
    preds = gru.train_model(model, norm_data, norm_data_y, seed=0)

    mae, mape, rmse = calculate_error(norm_data_y[2:], preds)
    print('mae: ', mae)
    print('mape: ', mape)
    print('rmse: ', rmse)

    # TEST MODEL ON UNSEEN TEST DATA ON FEBRUARYtest_model
    preds = gru.test_model(model, test_data, test_data_y)

    mae, mape, rmse = calculate_error(test_data_y[2:], preds)
    print('mae: ', mae)
    print('mape: ', mape)
    print('rmse: ', rmse)
    return preds

    # logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    #                     datefmt='%d-%m-%Y:%H:%M:%S',
    #                     level=logging.INFO,
    #                     filename='example.log')
    # logging.debug('This message should go to the log file')
    # logging.info('So should this')
    # logging.warning('And this, too')


def worker(seeds, mutex_w):

    proc_name = mp.Process.ident
    for seed in seeds:

        import time
        time.sleep(0.1)              # Computation intensive task

        mutex_w.acquire()
        print("====consume io====")  # IO resource
        mutex_w.release()


@log_time_taken
def exec_main():
    num_proc = mp.cpu_count()
    seeds = [np.arange(i, 100, num_proc) for i in range(num_proc)]
    mutex_w = mp.Lock()

    proc_list = list()
    for i in range(num_proc):
        t = mp.Process(target=worker, args=(seeds[i], mutex_w))
        proc_list.append(t)

    for t in proc_list:
        t.start()

    for t in proc_list:
        t.join()

    return True
