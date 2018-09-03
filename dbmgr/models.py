from django.db import models

import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer


class SANDP500(models.Model):
    date = models.DateField()
    open = models.DecimalField(max_digits=10, decimal_places=4)
    high = models.DecimalField(max_digits=10, decimal_places=4)
    low = models.DecimalField(max_digits=10, decimal_places=4)
    close = models.DecimalField(max_digits=10, decimal_places=4)
    volume = models.IntegerField()

    class Meta:
        ordering = ['-date']


class DJI(models.Model):
    date = models.DateField()
    open = models.DecimalField(max_digits=10, decimal_places=4)
    high = models.DecimalField(max_digits=10, decimal_places=4)
    low = models.DecimalField(max_digits=10, decimal_places=4)
    close = models.DecimalField(max_digits=10, decimal_places=4)
    volume = models.IntegerField()

    class Meta:
        ordering = ['-date']


class CredibleUSTwitterAccount(models.Model):
    uid = models.BigIntegerField()
    screen_name = models.TextField(max_length=80, unique=True)
    description = models.TextField(max_length=200, unique=True)

    def __str__(self):
        return self.screen_name


class AbstractNewsFeed(models.Model):
    created_datetime = models.DateTimeField()
    text = models.TextField(max_length=400)
    text2 = models.TextField(max_length=400, null=True)
    sentiment = models.DecimalField(max_digits=13, decimal_places=12, null=True)

    class Meta:
        abstract = True

    def replace_urls(self, input_text):
        """ remove urls """
        regex = r"https?\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)([a-zA-Z0-9\-\.\?\,\'\/\\\+&amp;%\$#_]*)?"
        return re.sub(regex, "", input_text, 0)

    def replace_common(self, input_text):
        """ replace common noun words """
        input_text = input_text.replace("U.S.", "US")

        # remove numbers
        regex = r"\b\d+(\.\d+)*\b"
        input_text = re.sub(regex, "", input_text, 0)

        return input_text

    def replace_stopwords(self, input_text):
        """ remove stopwords and punctuation, return list of words """
        tokenizer = RegexpTokenizer(r'\w+')
        word_list = tokenizer.tokenize(input_text)

        # remove stop words
        filtered_words = [word for word in word_list if word.lower() not in stopwords.words('english')]
        return filtered_words

    def replace_unimpt_words(self, word_list):

        # create list of unimportant words
        unimpt_words = []
        for unimpt_word in unimpt_words:
            word_list = [w for w in word_list if w != unimpt_word]

        return word_list

    def transform_text(self):
        """ update text2 with clean text """

        sno = SnowballStemmer('english')

        ttext = self.text
        ttext = self.replace_common(ttext)
        ttext = self.replace_urls(ttext)

        filtered_words = self.replace_stopwords(ttext)
        filtered_words = self.replace_unimpt_words(filtered_words)
        filtered_words = [sno.stem(word) for word in filtered_words]    # stemming

        self.text2 = " ".join(filtered_words)
        self.save()


class USTwitterNewsFeed(AbstractNewsFeed):
    feedid = models.BigIntegerField(unique=True)
    posted_by = models.ForeignKey(CredibleUSTwitterAccount, related_name="newsfeed")
    type = models.TextField(max_length=3, null=True)

    class Meta:
        ordering = ['feedid']


class NewspaperFeed(AbstractNewsFeed):
    source = models.TextField(max_length=400)
