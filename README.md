# Deep Learning with News Data

## Introduction
The purpose of this project is to experiment various deep learning models to predict stock prices. Deep learning models employed will attempt to incorporate financial news data into the model for accurate price prediction.

## Architecture
This project uses ORM feature of the Django framework as a standalone to query and save news sources to the database.

## How to use
- The `usnews.backup` file contains the database backup for news sources I used
- `sentiment\twitter_sentiment.py`
    * `def populate_twitter_account_to_db()`: Populate the database with credible twitter user account object defined in dbmgr/newsfeed/twitter_accts.txt 
    * `def populate_twitter_acct_tweets()`: Populate the database with tweets from credible user account
    * `def transform_twitter_feed()`: Clean twitter text of stop words, url links, etc.
- `workbench\test.py`
    * `def generate_twitter_feed_influence_score()`: perform scoring on twitter news feed in the database
    * `def execute_gru2_twitter_sentiment()`: train and test GRU prediction model

## Installation and setup guide
[Setting up the project](docs/INSTALL.md)
