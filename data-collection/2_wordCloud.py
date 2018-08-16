###################################################################################
## Description: Text analytics of the labelled mse data                         ###
## Status: WIP															        ###
###################################################################################
import json, time, datetime, sys
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import operator, string, re

# path to data folders
pathIn = '../data/build'
pathOut = '../data/results'

## choose weighting factor: default or tf-idf
# default: term frequency adjusted for document length 
# tf-idf: statistical measure to evaluate the importance of a word to a document in 
#         a collection, the importance increases proportionally to the number of times
#         a word appears in the document but is offset by the frequency of the word in the corpus
mode = 'default' #'tf-idf'

#========== reading data 
with open(pathIn+'/build_test.json', 'r') as read_file:
    df = pd.read_json(read_file)

egbotdo = dict()
egbotdont = dict()

docount = 0
dontcount = 0
# remove stopwords => nltk 
stop = set(stopwords.words('english'))

df['egbot_answer_body'] = df['egbot_answer_body'].apply(lambda x: [re.sub("[\.\t\,\:;\(\)\.\\\]", "", item, 0, 0) for item in string.split(x) if item.lower() not in stop])

 #========== constructing term frequency dictionaries 
for index, row in df.iterrows():
    words = row['egbot_answer_body']
    label = row['egbot_answer_label']
        
    if label:
        docount += 1 
    else:
        dontcount += 1

    for word in words:
        if label in "TRUE":
            if word in egbotdo:
                egbotdo[word] += 1
            else:
                egbotdo[word] = 1         
        else:
            if word in egbotdont:
                egbotdont[word] += 1
            else:
                egbotdont[word] = 1

# optional inverse document freq
if mode in 'tf-idf':
    appdo = dict()
    appdont = dict()
    totalWordsDo = 0
    totalWordsDont = 0

    for index, row in df.iterrows():
        words = row['egbot_answer_body']
        for word in egbotdo.keys():
            totalWordsDo += 1
            if word in words:
                if word in appdo:
                    appdo[word] += 1
                else:
                    appdo[word] = 1

        for word in egbotdont.keys():
            totalWordsDont += 1
            if word in words:
                if word in appdont:
                    appdont[word] += 1
                else:
                    appdont[word] = 1
                
    for word in egbotdo.keys():
        tf = egbotdo[word]/float(totalWordsDo)
        idf = np.log(df.shape[0]/(appdo[word]+appdont[word]))
        #egbotdo[word] = tf # relative term freq
        egbotdo[word] = int(tf*idf*100000) # multiplication by 100000 and conversion is required if we want word cloud to be able to display it 

    for word in egbotdont.keys():
        tf = egbotdont[word]/float(totalWordsDont)
        idf = np.log(df.shape[0]/(appdo[word]+appdont[word]))
        #egbotdont[word] = tf # relative term freq
        egbotdont[word] = int(tf*idf*100000) # multiplication by 100000 and conversion is required if we want word cloud to be able to display it 

# sort them based on frequency count, descending
freq_do = sorted(egbotdo.items(), key=operator.itemgetter(1), reverse=True)
freq_dont = sorted(egbotdont.items(), key=operator.itemgetter(1), reverse=True)

#=========== word cloud

# generate a word cloud image for egbotdo
wordcloud = WordCloud().generate_from_frequencies(egbotdo, max_font_size=40)

plt.figure()
plt.title('egbotdo')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# generate a word cloud image for egbotdont
wordcloud = WordCloud().generate_from_frequencies(egbotdont, max_font_size=40)

plt.figure()
plt.title('egbotdont')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# sort them based on frequency count, descending
freq_do = sorted(egbotdo.items(), key=operator.itemgetter(1), reverse=True)
freq_dont = sorted(egbotdont.items(), key=operator.itemgetter(1), reverse=True)

#=========== pie chart 
plt.figure()
count_Class=pd.value_counts(df['egbot_answer_label'], sort= True)
count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()