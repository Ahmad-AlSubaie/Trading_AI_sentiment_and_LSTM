import os
import subprocess
import tweepy
import re

# authenticate
apiKey = "BPwGcR04NbtsWwFJ6HCumgK08"
apiSecret = "BKkGwHt2BmEDMixFLBrLIwYyTOIuh6RjeMKhWBBBQwwyk8lfrD"
accessToken = "1357400772527415297-3gpsEPuSKCy7uMvDI66pvKlgciWOIh"
accessSecret = "qgOFFVopVZ8u4f0b9jplK4ucRZwEieqc3z4qR3EeyF0Wr"

authen = tweepy.OAuthHandler(apiKey, apiSecret)
authen.set_access_token(accessToken, accessSecret)
api = tweepy.API(authen)


def tweet_search(keyword, num):
    return tweepy.Cursor(api.search, q=keyword).items(num)


def save_tweets(tweets, filename):
    with open(".\\tweets\\" + filename, "w", encoding="utf-8") as tweetFile:
        for tweet in tweets:
            tweetFile.write(tweet.text)


def write_tweets_to_file(tweets, filename):
    with open(".\\tweets\\" + filename, "w", encoding="utf-8") as tweetFile:
        for tweet in tweets:
            tweetFile.write(tweet.text)
    with open("tweets.doclist", 'w', encoding="utf-8") as doc:
        doc.write(".\\tweets\\" + filename)


def get_backup_tweets(source, target):
    with open(".\\tweets\\" + source, "r", encoding="utf-8") as s:
        with open(".\\tweets\\" + target, "w", encoding="utf-8") as t:
            t.write(s.read())


def count_strong_pos(filename):
    with open(".\\tweets\\" + filename + "_auto_anns\subjclueslen1polar", "r", encoding="utf-8") as opinFile:
        text = opinFile.read()
        return text.count("strongpos")


def count_strong_neg(filename):
    with open(".\\tweets\\" + filename + "_auto_anns\\subjclueslen1polar", "r", encoding="utf-8") as opinFile:
        text = opinFile.read()
        return text.count("strongneg")


def count_weak_neg(filename):
    with open(".\\tweets\\" + filename + "_auto_anns\\subjclueslen1polar", "r", encoding="utf-8") as opinFile:
        text = opinFile.read()
        return text.count("weakneg")


def count_weak_pos(filename):
    with open(".\\tweets\\" + filename + "_auto_anns\\subjclueslen1polar", "r", encoding="utf-8") as opinFile:
        text = opinFile.read()
        return text.count("weakpos")


def count_neutral(filename):
    with open(".\\tweets\\" + filename + "_auto_anns\\subjclueslen1polar", "r", encoding="utf-8") as opinFile:
        text = opinFile.read()
        return text.count("neutral")


def count_all(filename):
    out = {'neutral': count_neutral(filename), 'weak_pos': count_weak_pos(filename),
           'weak_neg': count_weak_neg(filename), 'strong_neg': count_strong_neg(filename),
           'strong_pos': count_strong_pos(filename)}
    return out


def remove_untokenizable(readpath, writepath):
    with open(readpath, "r", encoding="utf-8") as f:
        with open(writepath, "w", encoding="utf-8") as w:
            txt = f.read()
            txt = re.sub(r'[^\x00-\x7F]', '', txt)
            w.write(txt)


def run_opinion_finder(doclist="tweets.doclist"):
    subprocess.run(
        'java -Xmx1g -classpath opinionfinderv2.0\\lib\\weka.jar;opinionfinderv2.0\\lib\\stanford-postagger.jar;opinionfinderv2.0\\opinionfinder.jar opin.main.RunOpinionFinder .\\' + doclist + ' -d -l opinionfinderv2.0\\lexicons',
        check=True, shell=True, capture_output=True)


# tw = tweepy.Cursor(api.search,lang="en", q="love").items(200)

# get_backup_tweets("sourcetesttweets", "testtweets")

# write_tweets_to_file(tw, "testtweets")
# save_tweets(tw, "sourcetesttweets")

# remove_untokenizable("./tweets/sourcetesttweets", "./tweets/testtweets")

# run_opinion_finder()


#print(count_all('testtweets'))
