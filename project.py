# -*- coding: utf-8 -*-
"""
Spyder Editor

author: adithya ganapathy (axg172330)
Title: Cosine similarity of two documents
"""
#import all necessary libraries
import pandas as pd
import numpy as np
import re
import math
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.data
import gensim
from matplotlib import pyplot
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings(action = 'ignore', category = UserWarning)
warnings.filterwarnings(action = 'ignore', category = DeprecationWarning)
np.seterr(divide='ignore', invalid='ignore')

#function to convert a document to sequence of words and remove stop words
#that returns a list of words
def wordlist(sen, remove_stopwords):
    text = BeautifulSoup(sen,"lxml").get_text()
    text = re.sub("[^a-zA-Z]"," ", text)
    words = text.lower().split()
    if remove_stopwords:
        stop = set(stopwords.words("english"))
        words = [w for w in words if not w in stop]
    return words

#function to split sentences into parsed sentences. Returns
#a list of sentences where each sentence is a list of words
def cleansentences( word, tokenizer, remove_stopwords):
    ini_sen = []
    word = word.encode('utf-8')
    ini_sen = tokenizer.tokenize(word.decode('utf-8').strip())
    sentences = []
    for sen in ini_sen:
        if(len(sen) > 0):
            sentences.append(wordlist(sen,remove_stopwords))
    return sentences

#function that calls wordlist function to generate seq of words
#value parameter determines the column
def cleanfile(words, value):
    sentences = []
    for word in words[value]:
        sentences.append(wordlist(word,True))
    return sentences

#function to average all the words in a paragraph
def makefeaturevecs(words, model, num_features):
    featurevec = np.zeros((num_features,), dtype="float32")
    num = 0
    wordcollections = set(model.wv.index2word)
    for word in words:
        if word in wordcollections:
            num = num + 1
            featurevec = np.add(featurevec, model[word])
    featurevec = np.divide(featurevec, num)
    return featurevec

#function to calculate the average feature vector for 
#each one and return a 2D numpy array
def getfeaturevecs(words, model, num_features):
    counter = 0
    featurevecs = np.zeros((len(words), num_features), dtype="float32")
    for word in words:
        featurevecs[counter] = makefeaturevecs(word, model, num_features)
        counter = counter + 1
    return featurevecs

#Read train file and create model
train = pd.read_csv('train.csv', header=0, delimiter=",")
print("Total number of documents:", train["product_title"].size)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = []
print("Cleaning and parsing training data for model creation...")
for word in train["product_title"]:
    sentences += cleansentences(word, tokenizer, True)
print("Parsing Completed...")
print("Training word2vec model...")
model = gensim.models.word2vec.Word2Vec(sentences, workers = 4, size = 300, min_count = 30, window = 10, sample = 1e-2, seed = 1)
model.init_sims(replace=True)
model.save("Feature_Model")
print("Model Created...")

#graphical representation of model's vocabulary
print("Vocab size = ", len(model.wv.vocab))
print("Graphical Representation of vocabulary:")
model = gensim.models.Word2Vec.load('Feature_Model')
X = model[model.wv.vocab]
pca = PCA(n_components = 2)
result = pca.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i,word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

#read test file and create feature vectors
test = pd.read_csv('test.csv', header = 0, delimiter = ",")
print("Creating feature vectors for product titles in test file...")
productdata = getfeaturevecs(cleanfile(test, "product_title"), model, 300)
print("Product title feature vectors extracted...")
print("Creating feature vectors for queries in test file...")
querydata = getfeaturevecs(cleanfile(test, "query"), model, 300)
print("Query feature vectors extracted...")
my_list = []

#determine cosine similarity of documents
print("Determining Cosine Similarity...")
qcount = 0
for i in querydata:
    count = 0
    for j in productdata:
        cos_sim = np.dot(i,j)/(np.linalg.norm(i)*np.linalg.norm(j))
        if math.isnan(cos_sim):
            cos_sim = 0.0
        my_list.append((test["query"][qcount],test["product_title"][count], cos_sim))
        count = count + 1
    qcount = qcount + 1
    if qcount == 50:
        break
print("Similarity calculation completed...")

#export it to a file
df = pd.DataFrame(my_list, columns = ["Query","Product_Title","Cosine_Similarity"])
df.to_csv("Output.csv",index=None)
print("Output file exported...")

#end of program
########################################################################