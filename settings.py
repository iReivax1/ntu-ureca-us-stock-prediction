import os

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '=re)#l0w$+h!y+%#@+5zl2rqvwhf8*(gpdv&rwyq&ebm^#vz2!'

# https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name=twitterapi&count=2

INSTALLED_APPS = [
    'dbmgr',
]

# Database
# https://docs.djangoproject.com/en/1.11/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'usnews',
        'USER': 'usnewsapp',
        'PASSWORD': 'qwe123qwe123',
        'HOST': '127.0.0.1',
        'PORT': '5432',
    }
}

TWITTER_OAUTH = {
    'consumer_key': os.environ.get('TWITTER_OAUTH_CONSUMER_KEY', 'your consumer key'),
    'consumer_secret': os.environ.get('TWITTER_OAUTH_CONSUMER_SECRET', 'your consumer secret key'),
    'access_token_key': os.environ.get('TWITTER_OAUTH_ACCESS_TOKEN', 'your access token key'),
    'access_token_secret': os.environ.get('TWITTER_OAUTH_ACCESS_SECRET', 'your access token secret key')
}
