# slim egbot data by saving only { "question": (was body_markdown)..., "answer": (was answers...nested body_markdown)... } from posts that have accepted answers
# you can also cap to a certain number of data this should take the MSE-full data that you have and slim it down to MSE-100 with just 100 data points)

import json, os

# assume in egbot/data-collection/json_analysis
buildPath = "../../data/build/"
assert os.listdir(buildPath), buildPath

# Find the fullname of the data files, e.g. MathStackExchangeAPI_Part_1_TimeStamps_1512760268_1535031491
def findFile(prefix):
    for file in os.listdir(buildPath):
        if file.startswith(prefix) and file.endswith(".json"):
            return file
    return ""

def trim(no, count):
    # find e.g. MathStackExchangeAPI_Part_1_TimeStamps_1512760268_1535031491.json (e.g. what the Zenodo fils are called)
    filename = findFile("MathStackExchangeAPI_Part_" + str(no))
    filepath = os.path.abspath(buildPath+filename)
    print("Opening " + filename+ " = "+filepath)
    with open(filepath) as f:
        data = json.load(f)
        print("Finished loading")

        slim = []
        for i in range(0, len(data)):
            if capped and count>=cap:
                break
            if("answers" in data[i].keys()):
                for j in range(0, len(data[i]["answers"])):
                    is_accepted = data[i]["answers"][j]["is_accepted"]
                    if (is_accepted):
                        temp = dict()
                        temp["question"] = data[i]["body_markdown"]
                        temp["answer"] = data[i]["answers"][j]["body_markdown"]
                        slim.append(temp)
                        count += 1                                
        if capped:
            filename = "MathStackExchangeAPI_" + str(cap) 
        outpath = os.path.abspath(buildPath + "slimmer/" + filename + ".json")
        print("Count: ", count)
        print("Saving ... to ", outpath)
        with open(outpath, 'w') as outfile:  
            json.dump(slim, outfile)
    return count

## main

# make the output directory
#os.mkdir(buildPath+"slimmer")

# this can be used if you want to trim to a dataset smaller than one of the 8 files; cap should be less than 50k (otherwise it would have to look at the next file)
# this will change the name of the file as well to MathStackExchangeAPI_*cap* e.g. MathStackExchangeAPI_500
capped = True
cap = 2000 
# tracker of data points count
count = 0

if (not capped):
    for no in range(1,9):
        count = trim(no, count)
else:
    for no in range(1,2):
        count = trim(no, count)

print("Done :)")

