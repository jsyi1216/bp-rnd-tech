# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:49:49 2019

@author: jsyi
"""

# Waitrose & Partners Food and Drink Report 2018-19
import time
import datetime
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import csv
import glob

# Read properties
def getProperties(propFile):
    props = []

    with open(propFile, newline='') as propFile:
        reader = csv.reader(propFile, delimiter=':')
        for row in reader:
            props.append(row[1])

    return props

# Read the file
def getText(input_path):
    sentences = []
    path = glob.glob(input_path+"/*.txt")
    
    for i in path :         
        f = open(i,"r", encoding='utf8')
        sentences += f.readlines()
    
    text = ''.join(sentences)

    return text

# Read stopwords
def getStopwords(stopWordsFile):
    stoplist = stopwords.words('english')
    arr = []

    with open(stopWordsFile , newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            arr.append(row[0])

    for s in [t for t in arr if t not in stoplist]:
        stoplist.append(s)
    
    return stoplist

# Read symonyms
def getSynonyms(synonymsFile):
    synonyms = dict()

    with open(synonymsFile, newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            synonyms[row[0]] = row[1] #?

    return synonyms

# generate stopwords
def getWords(text, stoplist, synonyms):
    list_of_words = [i.lower() for i in nltk.tokenize.wordpunct_tokenize(text) if i.lower() not in stoplist and i.isalpha()]

    for i, w in enumerate(list_of_words): 
        if synonyms.get(w) is not None:
            list_of_words[i] = synonyms.get(w)
    
    wordfreqdist = nltk.FreqDist(list_of_words)
    mostcommon = wordfreqdist.most_common(100)

    return mostcommon

# generate chat
def generateChart(mostcommon, resultsFolderPath):
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')

    x_labels = [val[0] for val in mostcommon]
    y_labels = [val[1] for val in mostcommon]
    
    plt.figure(figsize=(16, 8))
    ax = pd.Series(y_labels).plot(kind='bar')
    ax.set_xticklabels(x_labels)
    rects = ax.patches
    for rect, label in zip(rects, y_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, label, ha='center', va='bottom')
    fig = ax.get_figure()
    plt.gcf().canvas.set_window_title('Frequency chart')
    fig.savefig(resultsFolderPath+'/histogram_'+st+'.jpg')

# Generate a word cloud image
def generateWordcloud(text, stoplist, resultsFolderPath):
    wordcloud = WordCloud(stopwords=stoplist, background_color="white").generate_from_frequencies(text) #.generate(text)
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')

    fig = plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.gcf().canvas.set_window_title('Wordcloud')
    fig.savefig(resultsFolderPath+'/wordclould_'+st+'.jpg')

def main():
    props = getProperties("config.prop")

    try:
        inputPath = props[0]
        stopWordsFile = props[1]
        synonymsFile = props[2]
        resultsFolderPath = props[3]
    except:
        print("Please check the property file.")
        exit()


    
    text = getText(inputPath)
    stoplist = getStopwords(stopWordsFile)
    synonyms = getSynonyms(synonymsFile)
    mostcommon = getWords(text, stoplist, synonyms)
    generateChart(mostcommon, resultsFolderPath)
    generateWordcloud(dict(mostcommon), stoplist, resultsFolderPath)
    plt.show()
    print('Completed.')

if __name__ == "__main__":
    main()
