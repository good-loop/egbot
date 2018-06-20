###################################################################################
## Description: Process raw dedoose data into build data                        ###
## Status: WIP															        ###
###################################################################################

import sys, time
import pandas as pd

# script settings
iteration = 0 																				## UPDATE THIS: to improve data management

# path to data folder
path = '/home/irina/data'

if len(sys.argv) > 1:
    dataset = sys.argv[1]
    datacol = raw_input("What is the name of the DATA column? (press enter if not known)\n") 
    labelcol = raw_input("What is the name of the LABEL column? (press enter if not known) \n") 

    if datacol and labelcol:
        cols = [datacol,labelcol]
    else:
        cols = None
else:
    dataset = path + '/dedoose.csv'
    cols = ['Excerpt','Label'] #['Excerpt Copy','Meta Label']


# read data
df = pd.read_csv(dataset, usecols=cols, encoding='utf-8')

# exporting dataset to csv
try:
	df.to_csv(path + '/1_processRawData_%d_%d.csv' % (iteration, int(time.time())), sep=',', encoding='utf-8')
except UnicodeEncodeError as e: print(e)
