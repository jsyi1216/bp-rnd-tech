# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:10:21 2020

@author: dukkuk9
"""

# Waitrose & Partners Food and Drink Report 2018-19
import io
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from googletrans import Translator

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

import sys
import time
from PIL import Image
import numpy as np

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
    
    #for text
    sentences = []
    path = glob.glob(input_path+"/*.txt")
    
    for i in path :         
        f = open(i,"r", encoding='utf8')
        sentences += f.readlines()
    
    text = ''.join(sentences)
    
    #for pdf
    path = glob.glob(input_path+"/*.pdf")
    
    for i in path : 
        text = ''.join([text,extract_text_from_pdf(i)])
        
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
def generateWordcloud(text, stoplist, resultsFolderPath,maskImage):
    
    mask_shape = np.array(Image.open("data/"+maskImage+".png"))
    
    wordcloud = WordCloud(
    width = 700,
    height = 700,
    stopwords=stoplist,
    background_color="white",
    mask = mask_shape
    )
    
    wordcloud = wordcloud.generate_from_frequencies(text) #.generate(text)
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.gcf().canvas.set_window_title('Wordcloud')
    fig.savefig(resultsFolderPath+'/wordclould_'+st+'.jpg')


'''
-----------------------------------------
document summary modules begin from below 
-----------------------------------------
'''


def extract_text_from_pdf(pdf_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
 
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
 
        text = fake_file_handle.getvalue()
 
    # close open handles
    converter.close()
    fake_file_handle.close()
 
    if text:
        return text

def _create_frequency_table(text_string) -> dict:
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable

def _score_sentences(sentences, freqTable) -> dict:
    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

    return sentenceValue

def _find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = (sumValues / len(sentenceValue))

    return average

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def summary_doc(text, weight):
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)

    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, weight * threshold)

    return summary

def summary_file(text, weight):
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    
    summary = summary_doc(text, weight)    
    file2 = open("results/summary_"+st+".txt","w", encoding='utf8')
    file2.write(summary)
    file2.close()
    
    translator = Translator()
    translation = translator.translate(summary, dest='ko')
    file3 = open("results/translated_summary_"+st+".txt","w", encoding='utf8')
    file3.write(translation.text)
    file3.close()
    
'''
-----------------------------------------
              main section 
-----------------------------------------
'''


def main():
    props = getProperties("config.prop")
    mask_list = ['square','heart','bread','cloud','circle']
    isValid = 0
    try:
        inputPath = props[0]
        stopWordsFile = props[1]
        synonymsFile = props[2]
        resultsFolderPath = props[3]
        maskImage = props[4]
    except:
        print("Please check the property file.")
        exit()

   
    for i in mask_list :
        if maskImage == i: isValid = 1
    
    print("Wordcloud image : " + maskImage)
    
    if isValid == 0 :
        print("Error: Please check the mask name.")
        time.sleep(6)
        sys.exit()
            
    
    
    
    text = getText(inputPath)
    stoplist = getStopwords(stopWordsFile)
    synonyms = getSynonyms(synonymsFile)
    mostcommon = getWords(text, stoplist, synonyms)
    generateChart(mostcommon, resultsFolderPath)
    generateWordcloud(dict(mostcommon), stoplist, resultsFolderPath, maskImage)
    plt.show()
    print('Completed.')
    
    summary_file(text, 1.5)
    
if __name__ == "__main__":
    main()
