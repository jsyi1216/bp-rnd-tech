# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:36:40 2019

@author: jsyi
"""

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from collections import Counter
import networkx as nx
import re
from nltk.tokenize import RegexpTokenizer



filename="data/kalsec-2019.txt"
f=open(filename, "r", encoding='utf8')
sentences = f.readlines()
text = ''.join(sentences).lower().replace('\n',' ')

def wordnet_freq(text) : 
    
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    # Bring in the default English NLTK stop words
    stoplist = stopwords.words('english')
    # Define additional stopwords in a string
    additional_stopwords = ["Waitrose", "&", "Partners", "The", "food", "foods", "new", "trend", "trends", "flavors", "flavors", "consumers", "source"]

    # Split the the additional stopwords string on each word and then add
    # those words to the NLTK stopwords list
    stoplist += additional_stopwords


    list_of_words = [i.lower() for i in nltk.tokenize.wordpunct_tokenize(text) if i.lower() not in stoplist and i.isalpha()]
    wordfreqdist = nltk.FreqDist(list_of_words)
    mostcommon = wordfreqdist.most_common(50)

    return mostcommon,stoplist


def calculateWordset(sentences,stoplist):

    # Apply the stoplist to the text
    for i, c in enumerate(sentences):
        clean = [word for word in c.split() if word not in stoplist]
        sentence = ' '.join(word[0] for word in [[''.join(i)] for i in clean])
        sentence = re.sub('[,‘’=.#/?:$}]', '', sentence )   
        sentences[i] = sentence     


    bigrams = [b for l in sentences for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    bigram_counts = Counter(bigrams)
    bigram_df = pd.DataFrame(bigram_counts.items(), columns=['bigram', 'count'])

    return bigram_df

def generateDF(mostcommon,bigram_df):
    
    freqList=[]
    removedWordset=[]
    for m in mostcommon:
        freqList.append(m[0])

    for idx, bigram in enumerate(bigram_df['bigram']):
        if bigram[0] not in freqList and bigram[1] not in freqList:
            removedWordset.append(idx)

    df = bigram_df.drop(removedWordset, axis=0).sort_values(by ='count', ascending=False).reset_index(drop=True).head(30)
    return df
    
def generateWordNetwork(df):
    
    NODESIZE=1500
    FONTSIZE=14
    WITDH=1
    EDGECOLOR='grey'
    NODECOLOR='#A5DEE2'
    
    # Create dictionary of bigrams and their counts
    d = df.set_index('bigram').T.to_dict('records')
    # Create network plot 
    G = nx.Graph()
    # Create connections between nodes
    for k, v in d[0].items():
        G.add_edge(k[0], k[1], weight=(v * 5))
    #G.add_node("china", weight=100)
    fig, ax = plt.subplots(figsize=(20, 16))
    pos = nx.spring_layout(G, k=1)
    # Plot networks
    nx.draw_networkx(G, pos, node_size=NODESIZE,font_size=FONTSIZE, width=WITDH, edge_color=EDGECOLOR, node_color=NODECOLOR, with_labels = True, ax=ax)
    
    plt.show()
    fig.savefig('word-network.png')

  
'''
-----------------------------------------
              main section 
-----------------------------------------
'''


def main():
 
    
    # Open a file and read it into memory

    filename="data/kalsec-2019.txt"
    f=open(filename, "r", encoding='utf8')
    sentences = f.readlines()
    text = ''.join(sentences).lower().replace('\n',' ')


    
    
    mostcommon,stoplist = wordnet_freq(text)
    bigram_df = calculateWordset(sentences,stoplist)
    df = generateDF(mostcommon,bigram_df)
    generateWordNetwork(df)  
    
    
    print('Completed.')
    
    
if __name__ == "__main__":
    main()


