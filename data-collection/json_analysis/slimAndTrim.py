# slim egbot data by saving only { "question": ..., "answer": ... } from posts that have accepted answers

import json, os

buildPath = "../../data/build/"

for no in range(1,9):
    filename = "MathStackExchangeAPI_Part_" + str(no) + ".json"
    filepath = os.path.abspath(buildPath+filename)
    print("Opening " + filename)
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
                
        print("Saving ...")
        with open(os.path.abspath(buildPath + "slim/" + filename), 'w') as outfile:  
            json.dump(slim, outfile)

print("Done :)")