###################################################################################
## Description: Process raw dedoose data into build data                        ###
## Status: WIP															        ###
###################################################################################
import json, time, datetime, sys
import pandas as pd

# script settings
iteration = 0 																				## UPDATE THIS: to improve data management

# path to data folders
pathIn = '../data/raw'
pathOut = '../data/build'

if len(sys.argv) > 1:
    dataset = sys.argv[1]
    datacol = raw_input('What is the name of the DATA column? (press enter if not known)\n') 
    labelcol = raw_input('What is the name of the LABEL column? (press enter if not known) \n') 

    if datacol and labelcol:
        cols = [datacol,labelcol]
    else:
        #cols = None
        dataset = pathIn + '/d127+labelled.json'
        cols = ['egbot_answer_body','egbot_answer_id','egbot_answer_label'] 
else:
    dataset = pathIn + '/d127+labelled.json'
    cols = ['egbot_answer_body','egbot_answer_id','egbot_answer_label'] 

## read data
df = pd.DataFrame()
try:
    with open(dataset, 'r') as read_file:
        df = pd.read_json(read_file, encoding='utf-8')
    df = df[cols]
except IOError:
    print 'Fatal Error: Sorry, couldn\'t find ' + dataset
    sys.exit(0)

# exporting dataset
with open(pathOut + '/build_test.json', 'w') as outfile:  
    json.dump(df.to_dict(), outfile)
    #print outfile
