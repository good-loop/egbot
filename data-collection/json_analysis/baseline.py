
from sklearn.model_selection import KFold
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
import pandas as pd
import numpy as np
import sys

## script settings
iteration = 1 						
## path to data folder
buildPath = '/home/irina/data/build/slim/'

for no in range(1,9):
    filename = "MathStackExchangeAPI_Part_" + str(no) + ".json"
    filepath = os.path.abspath(buildPath+filename)
    print("Opening " + filename)
    with open(filepath) as f:
        data = json.load(f)
        print("Finished loading")

        slim = []
        for i in range(0, len(data)):
            data[i]["answer"]