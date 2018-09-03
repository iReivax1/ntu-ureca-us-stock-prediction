import dotenv
import getopt
import os
import sys

from django.core.wsgi import get_wsgi_application


def usage():
    print("Usage: python main.py [-h|help] [-f|fetch=<S&P500|DJI>]")


def main(argv):

    try:
        opts, remainders = getopt.getopt(argv, "hf:", ["help", "fetch=", "runmodel="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-f", "--fetch"):

            from dbmgr import db_ops
            stock_code = arg
            db_ops.fetch_and_save_stock_code_to_db(stock_code)
            sys.exit()
        elif opt in ("--runmodel", ):
            model_to_run = arg
            sys.exit()
        else:
            assert False, "unhandled option"

    for arg in remainders:
        if arg == 'runmodel':
            sys.exit()
        elif arg == 'trainmodel':
            sys.exit()
        elif arg == 'getnews':
            from sentiment.twitter_sentiment import populate_twitter_acct_tweets
            populate_twitter_acct_tweets()
            sys.exit()
        elif arg == 'newstext2':
            from sentiment.twitter_sentiment import transform_twitter_feed
            transform_twitter_feed()
            sys.exit()


if __name__ == "__main__":

    # READ SECRET SETTINGS
    dotenv.read_dotenv()
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
    application = get_wsgi_application()

    # from sentiment.twitter_sentiment import populate_twitter_account_to_db
    # populate_twitter_account_to_db()
    #
    # sys.exit()
    main(sys.argv[1:])
