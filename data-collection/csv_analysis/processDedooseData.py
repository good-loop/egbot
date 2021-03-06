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


## read data
try:
    df = pd.read_csv(dataset, usecols=cols, encoding='utf-8')
except IOError:
    print "Fatal Error: Sorry, couldn't find " + dataset
    sys.exit(0)

#df = df[:20] # smaller version for testing purposes 

df['ID'] = range(1,df.shape[0]+1) # id > 0, because of the principle of least astonishment 

# exporting dataset to csv
try:
    outputfile = path + '/1_processRawData_%d_%d.csv' % (iteration, int(time.time()))
    df.to_csv(outputfile, sep=',', index=False, encoding='utf-8')
except UnicodeEncodeError as e: print(e)

print outputfile
