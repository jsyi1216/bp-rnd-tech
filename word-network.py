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

# Bring in the default English NLTK stop words
stoplist = stopwords.words('english')

# Define additional stopwords in a string
additional_stopwords = ["Waitrose", "&", "Partners", "The", "food", "foods", "new", "trend", "trends", "flavors", "flavors", "consumers", "source"]

# Split the the additional stopwords string on each word and then add
# those words to the NLTK stopwords list
stoplist += additional_stopwords

# Open a file and read it into memory
filename="data/kalsec-2019.txt"
f=open(filename, "r", encoding='utf8')
sentences = f.readlines()
text = ''.join(sentences).lower().replace('\n',' ')
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

list_of_words = [i.lower() for i in nltk.tokenize.wordpunct_tokenize(text) if i.lower() not in stoplist and i.isalpha()]
wordfreqdist = nltk.FreqDist(list_of_words)
mostcommon = wordfreqdist.most_common(50)

x_labels = [val[0] for val in mostcommon]
y_labels = [val[1] for val in mostcommon]
plt.figure(figsize=(12, 6))
ax = pd.Series(y_labels).plot(kind='bar')
ax.set_xticklabels(x_labels)
rects = ax.patches
for rect, label in zip(rects, y_labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, label, ha='center', va='bottom')

# Apply the stoplist to the text
for i, c in enumerate(sentences):
    clean = [word for word in c.split() if word not in stoplist]
    sentence = ' '.join(word[0] for word in [[''.join(i)] for i in clean])
    sentence = sentence.replace(',','')
    sentence = sentence.replace('‘','')
    sentence = sentence.replace('’','')
    sentence = sentence.replace('[','')
    sentence = sentence.replace(']','')
#    sentence = sentence.replace(':','')
#    sentence = sentence.replace('?','')
    sentence = re.sub('[=.#/?:$}]', '', sentence )
    #print(sentence)
    sentences[i] = sentence


bigrams = [b for l in sentences for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
bigram_counts = Counter(bigrams)
bigram_counts.most_common(20)
bigram_counts.items()

bigram_df = pd.DataFrame(bigram_counts.items(), columns=['bigram', 'count'])

bigram_df['bigram']

t1=[]
t2=[]
for m in mostcommon:
    t1.append(m[0])

for idx, bigram in enumerate(bigram_df['bigram']):
    if bigram[0] == 'plant':
        print(bigram[0])
        print(bigram[1])
    if bigram[0] not in t1 and bigram[1] not in t1:
        t2.append(idx)

df = bigram_df.drop(t2, axis=0).sort_values(by ='count', ascending=False).reset_index(drop=True).head(30)
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
nx.draw_networkx(G, pos,
                 node_size=1500,
                 font_size=14,
                 width=1,
                 edge_color='grey',
                 node_color='#A5DEE2',
                 with_labels = True,
                 ax=ax)

# Create offset labels
#for key, value in pos.items():
#    x, y = value[0]+.045, value[1]+.045
#    ax.text(x, y,
#            s=key,
#            bbox=dict(facecolor='white', alpha=0),
#            horizontalalignment='center', fontsize=12)
    
plt.show()
fig.savefig('word-network.png')
