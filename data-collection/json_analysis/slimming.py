import json, os

for no in range(1,9):
    filename = "MathStackExchangeAPI_Part_" + str(no) + ".json"
    filepath = os.path.abspath("../data/build/"+filename)
    print("Opening " + filename)
    with open(filepath) as f:
        data = json.load(f)
        print("Finished loading")

        slim = []
        for i in range(0, len(data)):
            temp = dict()
            if("answers" in data[i].keys()):
                answers = []
                for j in range(0, len(data[i]["answers"])):
                    ans = dict()
                    ans["body_markdown"] = data[i]["answers"][j]["body_markdown"]
                    ans["is_accepted"] = data[i]["answers"][j]["is_accepted"]
                    answers.append(ans)
                temp["answers"] = answers
            temp["is_answered"] = data[i]["is_answered"]
            temp["answer_count"] = data[i]["answer_count"]
            temp["body_markdown"] = data[i]["body_markdown"]
            slim.append(temp)
        
        print("Saving ...")
        with open(os.path.abspath("../data/build/slim/"+filename), 'w') as outfile:  
            json.dump(slim, outfile)

print("Done :)")