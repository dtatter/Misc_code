#####Scrape tweets from Twitter to classify
##Example: 
#1) Scrape tweets from Twitter that have #bucs, #buccaneers, or #siegetheday in their text and are in English
#2) Save these tweets as a row to a .csv file
import twitterscraper
with open('nflbucs.csv','a',newline = '',encoding='utf-8') as fil:
    writer = csv.writer(fil)
    for tweet in twitterscraper.query_tweets("%23bucs%20OR%20%23buccaneers%20OR%20%23siegetheday%20lang%3Aen%20include%3Aretweets", 1000):
        writer.writerow(tweet)


####Train classifier based on tweet data
#0) Load data and setup
from nltk.corpus import twitter_samples

##take a sample of data
twitter_samples.strings('positive_tweets.json')[1]
twitter_samples.strings('negative_tweets.json')[1]

##create function word_feats() to turn string into a dictionary
def word_feats(words):
    return dict([(word, True) for word in words])
 
 
#1) a) Tokenize tweets from sample data
#b) Use word_feats() to create a dictionary out of the tokenized words
#c) Create list variable of positive and negative features using the dictionary from (b) and append 'pos' or 'neg'
import nltk
posfeats = [(word_feats(nltk.TweetTokenizer(preserve_case = False).tokenize(row)),'pos') for row in twitter_samples.strings('positive_tweets.json')]
len(posfeats) #check length - equivalent to number of tweets
negfeats = [(word_feats(nltk.TweetTokenizer(preserve_case = False).tokenize(row)),'neg') for row in twitter_samples.strings('negative_tweets.json')]
len(negfeats) #check length - equivalent to number of tweets


#2) Create training set and test set
poscutoff = round(len(posfeats) * 17/20)
negcutoff = round(len(negfeats) * 17/20)
trainfeats = posfeats[:poscutoff] + negfeats[:negcutoff] #will train classifier
testfeats = posfeats[poscutoff:] + negfeats[negcutoff:] #holdout data to test its accuracy
print('train %d instances, test on %d instances' % (len(trainfeats),len(testfeats)))

#3) Train classifier and test accuracy on holdout tweets
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util
classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:',nltk.classify.util.accuracy(classifier,testfeats)) #compares manual label from file with label done by classifier under classify_many()

##########################
## classify tweets ##
#0) Load data
import pandas
tweetdf = pandas.read_csv('bearsclean.csv') #insert CSV here - make sure it has header 'tweet' for column with users' tweets
list(tweetdf.columns.values) #check for name of columns (should see 'tweet')
tweetdf.tweet[100] #check to see tweet show up as a string
tweetcol = tweetdf.tweet #turn column into variable for ease
print(tweetcol[100]) #check to see tweet show up as a string
	
#1)Create dictionary of all words in the training set
##we want to do this because when we classify the nfl data, we need to see if words from that are part of the training data words
all_train_tweets = twitter_samples.strings('negative_tweets.json') #this creates a list of negative tweets and we append positive ones to it below
for postweet in	twitter_samples.strings('positive_tweets.json'):
	all_train_tweets.append(postweet)
all_words = set(word.lower() for passage in all_train_tweets for word in nltk.TweetTokenizer(preserve_case = False).tokenize((passage))) #we tokenize all of the tweets


#2)Tokenize the tweetcol strings into a dictionary of features with the True/False based on the dictionary from the training set
#see for more info: http://stackoverflow.com/questions/20827741/nltk-naivebayesclassifier-training-for-sentiment-analysis
test_sent_features2 = []
for tweet in tweetcol:
	test_sent_features2.append({word.lower(): (word in nltk.TweetTokenizer(preserve_case = False).tokenize(tweet.lower())) for word in all_words})

classifier.classify_many(test_sent_features2[0:15]) #see the classification for the first few

i = 0
dist = classifier.prob_classify_many(test_sent_features2[0:15]) #see probabalities of label - look for peculiarities (e.g. all 'pos' with prob. = 1.0)
for tweet in dist:
    print(tweetcol[i])
    i = i +1
    for label in tweet.samples():
        print("%s: %f" % (label, tweet.prob(label)))
		
nfl_classify = classifier.classify_many(test_sent_features2) #classify all tweets
print('Negative: %d, Total: %d, Fraction Negative: %f' % (nfl_classify.count('neg'), len(nfl_classify), nfl_classify.count('neg')/len(nfl_classify))) #how many and what percent were negative?


##Extra statistics on collected data
####turn tweetcol into one long string:
tweetcolstr = tweetcol.str.cat(sep=' ')

tt = nltk.TweetTokenizer(preserve_case = False).tokenize(tweetcolstr) #unigram statistics
nltk.FreqDist(tt).most_common(10)

ttbi = nltk.bigrams(tt) ##can do bigrams on existing tokenized data
nltk.FreqDist(ttbi).most_common(20)

tt4 = nltk.ngrams(tt,4) ##or just generally ngrams
nltk.FreqDist(tt4).most_common(20)
