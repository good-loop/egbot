# slim egbot data by saving only { "question": (was body_markdown)..., "answer": (was answers...nested body_markdown)... } from posts that have accepted answers

import json, os

# assume in egbot/data-collection/json_analysis
buildPath = "../../data/build/"
assert os.listdir(buildPath), buildPath

# Find the fullname of the data files, e.g. MathStackExchangeAPI_Part_1_TimeStamps_1512760268_1535031491
def findFile(prefix):
    for file in os.listdir(buildPath):
        if file.startswith(prefix) and file.endswith(".json"):
            return file
    return null

# make the output directory
os.mkdir(buildPath+"slim")

for no in range(1,9):
    # find e.g. MathStackExchangeAPI_Part_1_TimeStamps_1512760268_1535031491.json (e.g. what the Zenodo fils are called)
    filename = findFile("MathStackExchangeAPI_Part_" + str(no))
    filepath = os.path.abspath(buildPath+filename)
    print("Opening " + filename+ " = "+filepath)
    with open(filepath) as f:
        data = json.load(f)
        print("Finished loading")

        slim = []
        for i in range(0, len(data)):
            if("answers" in data[i].keys()):
                for j in range(0, len(data[i]["answers"])):
                    is_accepted = data[i]["answers"][j]["is_accepted"]
                    if (is_accepted):
                        temp = dict()
                        temp["question"] = data[i]["body_markdown"]
                        temp["answer"] = data[i]["answers"][j]["body_markdown"]
                        slim.append(temp)
                        
        outpath = os.path.abspath(buildPath + "slim/" + filename)
        print("Saving ... to ", outpath)
        with open(outpath, 'w') as outfile:  
            json.dump(slim, outfile)

print("Done :)")