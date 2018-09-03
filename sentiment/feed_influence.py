"""
SUMMARY:

Precondition: List of keywords, List of positive feeds, List of negative feeds

Postcondition: News feed sentiment value is updated with influence score
"""

from functools import reduce
from operator import mul
import re

# 1. collect 2 classes of twitter feed - feed during price rally, feed during
# price fall

# 2. extract dictionary of keywords and for each keyword calculate their
# posterior probability across all twitter feeds

# 3. calculate likelihood of a twitter feed to fall in positive class and
# negative class

# 4. calculate influence of twitter feed as "p(c1) / [p(c1) + p(c2)]"


class InfluenceCalculator():

    class KeywordItem():
        def __init__(self, keyword):
            self.keyword = keyword
            self.posClassKeywordCount = 0
            self.negClassKeywordCount = 0
            self.posClassPosteriorProbability = 0
            self.negClassPosteriorProbability = 0

    def __init__(self, keywords, posFeeds, negFeeds):
        self.keywords = [self.KeywordItem(kword) for kword in keywords]
        self.posFeeds = posFeeds
        self.negFeeds = negFeeds

        regex_keywords_pattern = r"\b\w+\b"

        posClassTotalKeyword = 0
        negClassTotalKeyword = 0

        # get number of keywords (TotalC) in positive feeds
        for posf in posFeeds:
            temp_text = posf.text2
            keyword_matches = re.finditer(regex_keywords_pattern, temp_text)
            for feedWordNum, _ in enumerate(keyword_matches):
                feedWordNum = feedWordNum + 1
            posClassTotalKeyword = posClassTotalKeyword + feedWordNum
        self.posClassTotalKeyword = posClassTotalKeyword

        # get number of keywords (TotalC) in negative feeds
        for negf in negFeeds:
            temp_text = negf.text2
            keyword_matches = re.finditer(regex_keywords_pattern, temp_text)
            for feedWordNum, _ in enumerate(keyword_matches):
                feedWordNum = feedWordNum + 1
            negClassTotalKeyword = negClassTotalKeyword + feedWordNum
        self.negClassTotalKeyword = negClassTotalKeyword

    def calculate_kword_count(self):
        for kword in self.keywords:

            posFeedClassTotalkeyword = 0
            negFeedClassTotalkeyword = 0
            regex_kword_pattern = r"\b" + kword.keyword + r"\b"

            # get number of occurance of keyword=kword for all positive feeds
            feedKwordNum = 0
            for posf in self.posFeeds:
                temp_text = posf.text2
                kword_matches = re.finditer(regex_kword_pattern, temp_text)
                for feedKwordNum, _ in enumerate(kword_matches):
                    feedKwordNum = feedKwordNum + 1
                posFeedClassTotalkeyword = posFeedClassTotalkeyword + feedKwordNum
            kword.posClassKeywordCount = posFeedClassTotalkeyword

            # get number of occurance of keyword=kword for all negative feeds
            feedKwordNum = 0
            for negf in self.negFeeds:
                temp_text = negf.text2
                kword_matches = re.finditer(regex_kword_pattern, temp_text)
                for feedKwordNum, _ in enumerate(kword_matches):
                    feedKwordNum = feedKwordNum + 1
                negFeedClassTotalkeyword = negFeedClassTotalkeyword + feedKwordNum
            kword.negClassKeywordCount = negFeedClassTotalkeyword

    def calculate_posterior_probability(self):
        for kword in self.keywords:
            kword.posClassPosteriorProbability = kword.posClassKeywordCount / self.posClassTotalKeyword
            kword.negClassPosteriorProbability = kword.negClassKeywordCount / self.negClassTotalKeyword

    def calculate_text_influence(self, input_text):
        word_list = input_text.split()
        keyword_list = [y for x in word_list for y in self.keywords if (x == y.keyword)]

        pos_likelihood = 0
        neg_likelihood = 0
        influence_score = 0

        if len(keyword_list) > 0:
            pos_likelihood = reduce(mul, [x.posClassPosteriorProbability for x in keyword_list])
            neg_likelihood = reduce(mul, [x.negClassPosteriorProbability for x in keyword_list])

        if (pos_likelihood + neg_likelihood) != 0:
            influence_score = pos_likelihood / (pos_likelihood + neg_likelihood)

        return influence_score
