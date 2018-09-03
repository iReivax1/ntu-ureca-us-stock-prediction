from .models import DJI, SANDP500

INDICES = {
    # https://stooq.com/q/d/l/?s=^dji&i=d
    'DJI': {'loc': 'dbmgr/data/indices/^dji_d.csv',
            'dbmodel': DJI
            },
    # https://stooq.com/q/d/l/?s=^spx&i=d
    'S&P500': {'loc': 'dbmgr/data/indices/^spx_d.csv',
               'dbmodel': SANDP500
               },
}

NEWSFEED = {
    'TWITTER': {'ACCOUNT_LIST': 'dbmgr/data/newsfeed/twitter_accts.txt'}
}

DICTIONARY = {
    'TWITTER': {'KEYWORD_LIST': 'dbmgr/data/dictionary/twitterfeed_keyword.txt'}
}
