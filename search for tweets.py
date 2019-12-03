import os
import tweepy
import pandas as pd
import GetOldTweets3 as got

list1=list()
max_tweets = 100

def tweetCriteria(start_date,end_date,movie_name):
    tweetCriteria = got.manager.TweetCriteria().setSince(start_date).setUntil(end_date).setQuerySearch(movie_name).setMaxTweets(max_tweets)
    for i in range(max_tweets):
        tweet = got.manager.TweetManager.getTweets(tweetCriteria)[i]
        a = (tweet.id, tweet.username, tweet.text, tweet.date)
        list1.append(a)
        df=pd.DataFrame(list1,columns=['TweetId','TweetUsername','TweetText','TweetDate'])
    return df


print(tweetCriteria("2018-06-22","2018-06-24","Hereditary").shape)
tweetCriteria("2018-07-05","2018-07-07","Hereditary")
tweetCriteria("2018-07-08","2018-07-10","Hereditary")
tweetCriteria("2018-07-20","2018-07-22","Hereditary")
print(tweetCriteria("2018-07-25","2018-07-27","Slender Man").shape)
tweetCriteria("2018-08-07","2018-08-09","Slender Man")
tweetCriteria("2018-08-10","2018-08-12","Slender Man")
tweetCriteria("2018-08-22","2018-08-24","Slender Man")


print(tweetCriteria("2018-03-25","2018-04-01","A Quiet Place").shape)
tweetCriteria("2018-04-02","2018-04-04","A Quiet Place")
tweetCriteria("2018-04-04","2018-04-06","A Quiet Place")
tweetCriteria("2018-04-23","2018-04-25","A Quiet Place")


print(tweetCriteria("2018-10-17","2018-10-19","Overlord").shape)
tweetCriteria("2018-10-31","2018-11-02","Overlord")
tweetCriteria("2018-11-03","2018-11-05","Overlord")
tweetCriteria("2018-11-15","2018-11-17","Overlord")

print(tweetCriteria("2018-08-30","2018-09-01","MandyHorrorMovie").shape)
tweetCriteria("2018-09-11","2018-09-13","MandyHorrorMovie")
tweetCriteria("2018-09-14","2018-09-16","MandyHorrorMovie")
tweetCriteria("2018-09-26","2018-09-28","MandyHorrorMovie")
print(tweetCriteria("2018-08-25","2018-08-27","The Nun").shape)
tweetCriteria("2018-09-04","2018-09-06","The Nun")
tweetCriteria("2018-09-07","2018-09-09","The Nun")
tweetCriteria("2018-09-19","2018-09-21","The Nun")
print(tweetCriteria("2018-08-27","2018-08-29","The Predator").shape)
tweetCriteria("2018-09-09","2018-09-11","The Predator")
tweetCriteria("2018-09-12","2018-09-14","The Predator")
tweetCriteria("2018-09-24","2018-09-26","The Predator")
print(tweetCriteria("2018-10-24","2018-10-25","Suspiria").shape)
tweetCriteria("2018-11-01","2018-11-02","Suspiria")
tweetCriteria("2018-11-02","2018-11-04","Suspiria")
tweets_info=tweetCriteria("2018-11-14","2018-11-16","Suspiria")

tweets_info.to_csv('tweets_info_final.csv', index=False)

