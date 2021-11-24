"""
Jason Yin and Ahmad Alsubaie
Twitter sentiment analysis, python code, cs499nlp
remember to do this to install libraries:
pip install textblob and pip install tweepy
"""
import re #regular expression
import tweepy #python client for official twitter api, install with 'pip install tweepy'
import textblob #process text library, use 'pip install textblob' and 'python -m textblob.download_corpora'
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import nltk
import pycountry
import string
from textblob import TextBlob
from tweepy import OAuthHandler #authentication

#authenticate
apiKey = "krqTvCVlFX1Fs3lMJWvSZrxEq"
apiSecret = "GZsAnHmAhg7bWOGRq6KD7U2vX1ISP1QuigkBLR5ov5u2Bh21y7"
accessToken = "1357400772527415297-3gpsEPuSKCy7uMvDI66pvKlgciWOIh"
accessSecret = "qgOFFVopVZ8u4f0b9jplK4ucRZwEieqc3z4qR3EeyF0Wr"

authen = tweepy.OAuthHandler(apiKey, apiSecret)
authen.set_access_token(accessToken, accessSecret)
api = tweepy.API(authen)

#searching Twitter
positive = 0
negative = 0
neutral = 0
searchword = input("Enter hashtag or keyword for search: ")
numTweets = int(input("How many tweets would you like to analyze?: "))
tweets = tweepy.Cursor(api.search,q=searchword).items(numTweets)
tweet_lis = []
pos_lis = []
neg_lis = []
neutral_lis = []
polarity = 0


def percent(numer,denom):
    return 100*float(numer)/float(denom)
for tweet in tweets:
    print(tweet.text) #prints tweet text
    tweet_lis.append(tweet.text)

    anal = TextBlob(tweet.text)#analysis part
    polscore = SentimentIntesityAnalyzer().polarity_scores(tweet.text)
    neut = polscore["neu"]
    pos = polscore["pos"]
    nega = polscore["neg"]
    c = score['compound']
    polarity = polarity + anal.sentiment.polarity

    if nega > pos:
        neg_lis.append(tweet.text)
        negative=negative+1
    elif pos > nega:
        pos_lis.append(tweet.text)
        positive=positive+1
    elif nega == pos:
        neutral_lis.append(tweet.text)
        neutral=neutral+1
positive = percentage(positive,numTweets)
neutral = percentage(neutral,numTweets)
negative = percentage(negative,numTweets)
polarity = percentage(polarity,numTweets)

#total, pos, neg, neutral
tweet_lis = pd.DataFrame(tweet_lis)
neg_lis = pd.DataFrame(neg_lis)
neutral_lis = pd.DataFrame(neutral_lis)
pos_lis = pd.DataFrame(pos_lis)

#printing
print("Total tweets: ",len(tweet_lis))
print("Positive tweets: ",len(pos_lis))
print("Neutral tweets: ",len(neutral_lis))
print("Negative tweets: ",len(neg_lis))

