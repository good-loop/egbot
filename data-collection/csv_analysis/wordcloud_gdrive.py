import json, time, datetime, sys
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import operator, string, re

egbotdo = dict()
egbotdont = dict()
appDoAndDont = dict()
totalWordsDo = 0
totalWordsDont = 0
docount = 0
dontcount = 0

# remove stopwords => nltk 
stop = set(stopwords.words('english'))

#========== constructing term frequency dictionaries 
for index, row in df.iterrows():
    words = row['egbot_answer_body']
    label = row['egbot_answer_label']
        
    if label:
        docount += 1 
    else:
        dontcount += 1

    for word in words:
        if word in appDoAndDont:
            appDoAndDont[word] += 1
        else:
            appDoAndDont[word] = 1  
        if label in "TRUE":
            totalWordsDo += 1
            if word in egbotdo:
                egbotdo[word] += 1
            else:
                egbotdo[word] = 1         
        else:
            totalWordsDont += 1
            if word in egbotdont:
                egbotdont[word] += 1
            else:
                egbotdont[word] = 1
