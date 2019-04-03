
# Welcome to the EgBot Example Generator Project

Code structure: follows the Good-Loop standard setup. Which is a js/react UI, and a Java server.

## Data

See the [EgBot Maths Q&A blog post](https://platypusinnovation.blogspot.com/2018/10/egbot-maths-q-dataset.html)

The data is collected by Irina from mathstackexchange using Python scripts in the data-collection folder.

You are advised to get some snapshot data, rather than run these yourself.

Data should then be kept in the `data\raw` folder.

In G-Drive: "EgBot Data" (which has manually labeled data):
https://drive.google.com/drive/u/0/folders/1K6US9Zy3WCJN0HKlCOejoT6cTnwBAlz-

We also have a Zenodo folder (which has the latest MathStackExchange dataset)
https://zenodo.org/communities/egbot?page=1&size=20

## Running Egbot Application Locally

### Requirements

* python 3.5 -- e.g. add this to .bashrc to avoid Python 2
	alias python='python3.5'
* pip3 8.1
* java 1.8
* com.winterwell.maths.stats.distributions.cond (restricted library)

Install required python packages using pip:

* tensorflow 
* keras 
* numpy
* OLD not needed anymore: jep

Install required Java jars using bob

Set the environment variables:
 
`LD_PRELOAD` path to your libpythonVERSION.so
`LD_LIBRARY_PATH` path to the jepVERSION.so 

e.g. 
	export LD_PRELOAD="$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libpython3.5m.so"
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/python3.5/dist-packages:/home/irina/.local/lib/python3.5/site-packages"

## Training Models (Optional, recommended using default trained model found in data/models/final/v1)

### Requirements

python 3.5
pip 18.0                   
cuda 9.0
cudnn 7.0
(install required python packages listed in data-collection/train-lstm-requirements.txt using pip)


### Test your setup

This is a test because only takes a small subset of the full data set:

1. cd to data-collection/json-analysis and run python slimAndTrim.py (this should take the MSE-full data that you have and slim it down to MSE-100 with just 100 data points)
2. you need to build egbot so run bob (because you'll need to make sure to have specific tf library)
3. run EvaluationTest.java and it should run the Markov evaluation (first training on MSE-100 data and then evaluating using the MSE-20 data)
4. the console should specify where it saved the results
5. comment out test100Markov in EvaluationTest and uncomment test100LTSM
6. cd to data-collection/build\_graph and run python createLSTMGraphTF.py (this will create the graph needed to run LSTM training)
7. run EvaluationTest.java again to evaluate LSTM (first it will train and then evaluate)
8. the console should specify where it saved the results

### Running the code (documentation is WIP)

1. you need to build egbot so run bob (because you'll need to make sure to have specific tf library)
2. run EvaluationTest.java and it should run the Markov evaluation 
3. the console should specify where it saved the results
4. comment out testFullMarkov in EvaluationTest and uncomment testFullLTSM
5. cd to data-collection/build\_graph and run python createLSTMGraphTF.py (this will create the graph needed to run LSTM training)
6. run EvaluationTest.java again to evaluate LSTM (first it will train and then evaluate)
7. the console should specify where it saved the results


