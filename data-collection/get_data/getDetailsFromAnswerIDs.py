# json script to prepend egbot data with index no
# maybe use mse api to easily get some data to test with
# test a small sample with elastic search to see whether it works

###########################################################################
## Description: Script to prepend json data with index for ES bulk add 	###
## Status: WIP															###
###########################################################################

# -*- coding: utf-8 -*-
# import required libraries
import requests, json, time, datetime, csv, sys
from pandas.io.json import json_normalize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import operator


# path to data folder
path = '/home/irina/data'

df = pd.read_csv(path + '/egbotdoTest.csv', encoding='utf-8')

egbotdo = dict()

#========== constructing frequency dictionaries
for index, row in df.iterrows():

    print row

    # requesting answer details    
    request_link = 'http://api.stackexchange.com/2.2/questions/?key=D7oawZM*bnJwSm6e4UC2jA((&order=desc&sort=creation&pagesize=10&site=math.stackexchange&filter=!asyat-glvTDN7Q*KS8FC0X2Ds8E427nbJlZsxliDugZwP6._EWQ0H)6SoS2c'

    # make api request 
    r = requests.get(request_link)
    temp = json.loads(r.text)
    while 'error_id' in temp.keys():
        print "Sleeping for 60 seconds ..."
        time.sleep(60)
        r = requests.get(request_link)
        temp = json.loads(r.text)	
    data = temp

    if "items" in data.keys():
        if len(data["items"]) > 0:
            for question in data["items"]:
                print question
                #final.append(question)
        else:
            print "Error: Couldn't find questions in this response"
    else:
        print "===========API-ERROR============="
        print data