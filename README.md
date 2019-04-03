
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
cuda 8.0
cudnn
(install required python packages listed in data-collection/train-lstm-requirements.txt using pip)

