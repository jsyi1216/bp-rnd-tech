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
    
    text = ''.join(sentences) #list -> string

    return text

# Read stopwords
def getStopwords(stopWordsFile): # Stopwords: 빈도수를 계산할 때 제외되는 단어 
    stoplist = stopwords.words('english')#stopwords.words("english")는 NLTK가 정의한 영어 불용어 리스트를 리턴
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
#i.lower가 stoplist에 없으며, alpha이라면 text의 token값을 list_of_words에 저장
    for i, w in enumerate(list_of_words): #enumerate : index값을 i에 넣어서 반환하는 for
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

    # Display the generated image:
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear') #nearist,none등 옵션 있고 pixel softner 정도 생각
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

    #print(path)
    
    text = getText(inputPath) #inputFile
    stoplist = getStopwords(stopWordsFile)
    synonyms = getSynonyms(synonymsFile)
    mostcommon = getWords(text, stoplist, synonyms)
    generateChart(mostcommon, resultsFolderPath)
    generateWordcloud(dict(mostcommon), stoplist, resultsFolderPath)
    plt.show()
    print('Completed.')

if __name__ == "__main__":
    main()
