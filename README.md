TODO - please add notes here as you go

# Welcome to the Example Generator Project

Code structure: follows the Good-Loop standard setup. Which is a js/react UI, and a Java server.

## Data

The data is collected by Irina from mathstackexchange using Python scripts in the data-collection folder.

You are advised to get some snapshot data, rather than run these yourself.

Data should then be kept in the `data\raw` folder.

In G-Drive: "EgBot Data" (which has manually labeled data):
https://drive.google.com/drive/u/0/folders/1K6US9Zy3WCJN0HKlCOejoT6cTnwBAlz-

We also have a Zenodo folder (which has the latest MathStackExchange dataset)
https://zenodo.org/communities/egbot?page=1&size=20

## Running Egbot Application Locally

### Requirements
java 1.8.0_181
jep 3.8.2                  
TODO: add rest of application requirements

## Training Models (Optional, recommended using default trained model found in data/models/final/v1)

### Requirements
python 3.5
pip 18.0                   
cuda 8.0
cudnn
(install required python packages listed in data-collection/train-lstm-requirements.txt)

