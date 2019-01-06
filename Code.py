import twitter
import tweepy
import time
import csv
import re
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

api = twitter.Api(consumer_key=' ',
                 consumer_secret=' ',
                 access_token_key=' ',
                 access_token_secret=' ')

print(api.VerifyCredentials())

#Function to accept a search term and then fetch the tweets for that term

def createTestData(search_string):
    try:
        tweets_fetched=api.GetSearch(search_string, count=100)
        # Return a list with twitter.Status objects. These have attributes for text, hashtags of tweets.
        print ("Fetched "+str(len(tweets_fetched))+" tweets with the term "+search_string+"!!")
        # We will fetch only the text for each of the tweets, and since these don't have labels yet, so empty label
        return [{"text":status.text,"label":None} for status in tweets_fetched]
    except:
        print ("Sorry there was an error!")
        return None

search_string=input("What are we searching for today?")
testData=createTestData(search_string)


testData[0:9]

def createTrainingCorpus(corpusFile,tweetDataFile):
    corpus=[]
    with open(corpusFile,'rb') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2],"label":row[1],"topic":row[0]})


    rate_limit=180
    sleep_time=900/180

    trainingData=[]
    for tweet in corpus:
        try:
            status=api.GetStatus(tweet["tweet_id"])
            #Returns a twitter.Status object
            print ("Tweet fetched" + status.text)
            tweet["text"]=status.text
            #tweet is a dictionary which already has tweet_id and label (positive/negative/neutral)
            trainingData.append(tweet)
            time.sleep(sleep_time) # to avoid being rate limited
        except:
            continue
    with open(tweetDataFile,'wb') as csvfile:
        linewriter=csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in trainingData:
            try:
                linewriter.writerow([tweet["tweet_id"],tweet["text"],tweet["label"],tweet["topic"]])
            except (Exception,e):
                print (e)
    return trainingData



# For small data set

def createLimitedTrainingCorpus(corpusFile,tweetDataFile):
    corpus=[]
    with open(corpusFile,'rb') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2],"label":row[1],"topic":row[0]})

    trainingData=[]
    for label in ["positive","negative"]:
        i=1
        for tweet in corpus:
            if tweet["label"]==label and i<=50:
                try:
                    status=api.GetStatus(tweet["tweet_id"])
                    #Returns a twitter.Status object
                    print ("Tweet fetched" + status.text)
                    tweet["text"]=status.text
                    #tweet is a dictionary which already has tweet_id and label (positive/negative/neutral)
                    trainingData.append(tweet)
                    i=i+1
                except (Exception, e):
                    print (e)

    with open(tweetDataFile,'wb') as csvfile:
        linewriter=csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in trainingData:
            try:
                linewriter.writerow([tweet["tweet_id"],tweet["text"],tweet["label"],tweet["topic"]])
            except (Exception,e):
                print (e)
    return trainingData

corpusFile="/Users/AMAN/Desktop/corpus.csv"
tweetDataFile="/Users/AMAN/Desktop/tweets.csv"

trainingData=createLimitedTrainingCorpus(corpusFile,tweetDataFile)

#Preprocessing of tweets

class PreProcessTweets:
    def __init__(self):
        self._stopwords=set(stopwords.words('english')+list(punctuation)+['AT_USER','URL'])

    def processTweets(self,list_of_tweets):
        # The list of tweets is a list of dictionaries which should have the keys, "text" and "label"
        processedTweets=[]
        # This list will be a list of tuples. Each tuple is a tweet which is a list of words and its label
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets

    def _processTweet(self,tweet):
        # 1. Convert to lower case
        tweet=tweet.lower()
        # 2. Replace links with the word URL
        tweet=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
        # 3. Replace @username with "AT_USER"
        tweet=re.sub('@[^\s]+','AT_USER',tweet)
        # 4. Replace #word with word
        tweet=re.sub(r'#([^\s]+)',r'\1',tweet)
        tweet=word_tokenize(tweet)
        # This tokenizes the tweet into a list of words
        return [word for word in tweet if word not in self._stopwords]

tweetProcessor=PreProcessTweets()
ppTrainingData=tweetProcessor.processTweets(trainingData)
ppTestData=tweetProcessor.processTweets(testData)


# First build a vocabulary
def buildVocabulary(ppTrainingData):
    all_words=[]
    for (words,sentiment) in ppTrainingData:
        all_words.extend(words)
    # This will give us a list in which all the words in all the tweets are present
    wordlist=nltk.FreqDist(all_words)
    # This will create a dictionary with each word and its frequency
    word_features=wordlist.keys()
    # This will return the unique list of words in the corpus
    return word_features


def extract_features(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
        # This will give us a dictionary , with keys like 'contains word1' and 'contains word2' and values as true false.
    return features

# Now we can extract the features and train the classifier
word_features = buildVocabulary(ppTrainingData)
trainingFeatures=nltk.classify.apply_features(extract_features,ppTrainingData)

NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)

svmTrainingData=[' '.join(tweet[0]) for tweet in ppTrainingData]
# Creates sentences out of the lists of words

vectorizer=CountVectorizer(min_df=1)
X=vectorizer.fit_transform(svmTrainingData).toarray()
# We now have a term document matrix
vocabulary=vectorizer.get_feature_names()

#Using sentiwordnet
swn_weights=[]

for word in vocabulary:
    try:
        synset=list(swn.senti_synsets(word))
        # use the first synset only to compute the score, as this represents the most common usage of that word
        common_meaning =synset[0]
        # If the pos_Score is greater, use that as the weight, if neg_score is greater, use -neg_score as the weight
        if common_meaning.pos_score()>common_meaning.neg_score():
            weight=common_meaning.pos_score()
        elif common_meaning.pos_score()<common_meaning.neg_score():
            weight=-common_meaning.neg_score()
        else:
            weight=0
    except:
        weight=0
    swn_weights.append(weight)


swn_X=[]
for row in X:
    swn_X.append(np.multiply(row,np.array(swn_weights)))
# Convert the list to a numpy array
swn_X=np.vstack(swn_X)


labels_to_array={"positive":1,"negative":2}
labels=[labels_to_array[tweet[1]] for tweet in ppTrainingData]
y=np.array(labels)


SVMClassifier=SVC()
SVMClassifier.fit(swn_X,y)


NBResultLabels=[NBayesClassifier.classify(extract_features(tweet[0])) for tweet in ppTestData]

# Now SVM
SVMResultLabels=[]
for tweet in ppTestData:
    tweet_sentence=' '.join(tweet[0])
    svmFeatures=np.multiply(vectorizer.transform([tweet_sentence]).toarray(),np.array(swn_weights))
    SVMResultLabels.append(SVMClassifier.predict(svmFeatures)[0])
    # predict() returns  a list of numpy arrays, get the first element of the first array
    # there is only 1 element and array


if NBResultLabels.count('positive')>NBResultLabels.count('negative'):
    print ("NB Result Positive Sentiment" + str(100*NBResultLabels.count('positive')/len(NBResultLabels))+"%")
else:
    print ("NB Result Negative Sentiment" + str(100*NBResultLabels.count('negative')/len(NBResultLabels))+"%")




if SVMResultLabels.count(1)>SVMResultLabels.count(2):
    print ("SVM Result Positive Sentiment" + str(100*SVMResultLabels.count(1)/len(SVMResultLabels))+"%")
else:
    print ("SVM Result Negative Sentiment" + str(100*SVMResultLabels.count(2)/len(SVMResultLabels))+"%")


testData[0:10]

NBResultLabels[0:10]

SVMResultLabels[0:10]
