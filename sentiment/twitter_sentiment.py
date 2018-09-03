import itertools
from utils import write_and_restart_line

from datetime import date, datetime, timedelta
from decorators import deprecated
from django.utils import timezone
from django.conf import settings
from django.db import IntegrityError

from dbmgr.data_config import NEWSFEED
from dbmgr.models import CredibleUSTwitterAccount, USTwitterNewsFeed

import twitter
from twitter.error import TwitterError

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def populate_twitter_account_to_db():
    """
    Create database of credible twitter user account if does not already
    exists
    """
    api = twitter.Api(**settings.TWITTER_OAUTH, sleep_on_rate_limit=False)
    with open(NEWSFEED['TWITTER']['ACCOUNT_LIST'], 'r') as f:
        lines = f.readlines()
        for l in lines:
            screen_name = l.strip()

            if CredibleUSTwitterAccount.objects.filter(screen_name=screen_name).exists():
                continue

            try:
                twitteruser = api.GetUser(screen_name=screen_name)
                CredibleUSTwitterAccount.objects.create(screen_name=twitteruser.screen_name,
                                                        uid=twitteruser.id,
                                                        description=twitteruser.description)
            except TwitterError as e:
                print(e.message)


def test_twitter_api():
    api = twitter.Api(**settings.TWITTER_OAUTH, sleep_on_rate_limit=False)
    statuses = api.GetUserTimeline(screen_name='ReutersUS', include_rts=False)
    print(statuses[0])


# issues: standard search api only allow past 7 days search
def populate_twitter_acct_tweets_by_date():
    """
    Populate tweets of user account already in the database
    for specified from_date to until_date
    """
    api = twitter.Api(**settings.TWITTER_OAUTH, sleep_on_rate_limit=False)
    twitter_accts = CredibleUSTwitterAccount.objects.all()

    for acct in twitter_accts:
        results = api.GetSearch(raw_query="l=&q=from%3AReutersUS%20since%3A2017-12-01%20until%3A2017-12-02&src=typd")


@deprecated
def populate_twitter_acct_tweets(retrieve_until_dt=datetime.now(tz=timezone.utc) - timedelta(days=60)):
    """
    Populate tweets of user account already in the database
    - function not suitable for use due to retrieve limit of 3200 twitter feeds
    """
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    api = twitter.Api(**settings.TWITTER_OAUTH, sleep_on_rate_limit=False)
    twitter_accts = CredibleUSTwitterAccount.objects.all()

    while 1:
        for acct in twitter_accts:
            # acct_oldest_tweet = USTwitterNewsFeed.objects.filter(posted_by=acct).first()
            acct_oldest_tweet = USTwitterNewsFeed.objects.filter(posted_by=acct, created_datetime__gte=date(2018, 2, 7)).first()

            max_id = None
            if acct_oldest_tweet is not None:
                max_id = acct_oldest_tweet.feedid - 1

            # do api call 15 for each account times due to twitter rate limit
            for _ in range(15):
                feed_created_dt = None
                try:
                    statuses = api.GetUserTimeline(screen_name=acct.screen_name, include_rts=False, max_id=max_id)
                    for s in statuses:
                        write_and_restart_line(next(spinner))
                        created_feed = USTwitterNewsFeed.objects.create(posted_by=acct,
                                                                        created_datetime=datetime.strptime(s.created_at, '%a %b %d %X %z %Y'),
                                                                        text=s.text,
                                                                        feedid=s.id)
                        max_id = created_feed.feedid - 1
                        feed_created_dt = created_feed.created_datetime
                except TwitterError as e:
                    print(e.message)
                except IntegrityError as e:
                    print('integrity error')
                    break

                # only retrieve until last status created datetime earlier than retrieve until
                # if (feed_created_dt is None) or (feed_created_dt < retrieve_until_dt):
                #     break


def transform_twitter_feed():

    no_text2_twitter_feed = USTwitterNewsFeed.objects.filter(text2=None)
    for twitter_feed in no_text2_twitter_feed:
        twitter_feed.transform_text()
