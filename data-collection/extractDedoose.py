###################################################################################
## Description: Text analytics of the labelled dedoose data                     ###
## Status: WIP															        ###
###################################################################################

from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import operator

#========== reading data 
df = pd.read_csv('data/dedoose.csv', encoding='utf-8')

egbotdo = dict()
egbotdont = dict()

docount = 0
dontcount = 0

#========== constructing frequency dictionaries
for index, row in df.iterrows():
    words = row['Excerpt'].lower().split(' ')
    label = row['Label']
    
    if label in 'egbotdont':
        dontcount += 1
    if label in 'egbotdo':
        docount += 1 

    for word in words:
        if label in 'egbotdont':
            if word in egbotdont:
                egbotdont[word] += 1
            else:
                egbotdont[word] = 1
        if label in 'egbotdo':
            if word in egbotdo:
                egbotdo[word] += 1
            else:
                egbotdo[word] = 1

# sort them based on frequency count, descending
freq_do = sorted(egbotdo.items(), key=operator.itemgetter(1), reverse=True)
freq_dont = sorted(egbotdont.items(), key=operator.itemgetter(1), reverse=True)

#=========== word cloud

# remove stopwords => nltk 
stop = set(stopwords.words('english'))

egbotdo_filtered = dict()
egbotdont_filtered = dict()

for word in egbotdo:
    if word not in stop:
        egbotdo_filtered[word] = egbotdo[word]
for word in egbotdont:
    if word not in stop:
        egbotdont_filtered[word] = egbotdont[word]

# generate a word cloud image for egbotdo
wordcloud = WordCloud().generate_from_frequencies(egbotdo_filtered, max_font_size=40)

plt.figure()
plt.title('egbotdo')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# generate a word cloud image for egbotdont
wordcloud = WordCloud().generate_from_frequencies(egbotdont_filtered, max_font_size=40)

plt.figure()
plt.title('egbotdont')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')

# sort them based on frequency count, descending
freq_do = sorted(egbotdo_filtered.items(), key=operator.itemgetter(1), reverse=True)
freq_dont = sorted(egbotdont_filtered.items(), key=operator.itemgetter(1), reverse=True)

#=========== pie chart 
plt.figure()
count_Class=pd.value_counts(df['Label'], sort= True)
count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()