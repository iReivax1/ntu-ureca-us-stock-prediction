import newspaper
from datetime import datetime


def populate_newspaper_feed():

    article_list = []
    oldest_date = datetime.today()
    oldest_article_url = None

    # cnn_paper = newspaper.build('http://money.cnn.com', language='en', is_memo=False)
    cnn_paper = newspaper.build('http://money.cnn.com', language='en', memoize_articles=False)

    for i in range(2):

        if oldest_article_url is not None:
            cnn_paper = newspaper.build(oldest_article_url, language='en', memoize_articles=False)

        for article in cnn_paper.articles:
            article.download()
            article.parse()
            if article.publish_date <= oldest_date:
                oldest_date = article.publish_date
                oldest_article_url = article.url
                article_list.append(article)
            # print(article.url)

    with open('dbmgr/data/newsfeed/articles.txt', 'wb') as f:
        for art in article_list:
            f.write(art.url + "\n")

    # art = cnn_paper.articles[0]
    # art.download()
    # art.parse()
    # art.publish_date
