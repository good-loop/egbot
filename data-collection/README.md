
## The following scripts analyse the test data:

0_turnLabelledIDsIntoRawData.py : gathers the data required for the next scripts based on data from /data/raw/ids+labels.json

1_processRawData.py : processes the contents of /data/raw/d127+labelled.json

2_wordCloud.py : creates a word cloud with the contents of /data/build/build_test.json


The scripts can be run through piping, if so desired (the output file of the first script will be used as the input of the next)

```
python 0_turnLabelledIDsIntoRawData.py | python 1_processRawData.py | python 2_wordCloud.py 
```
