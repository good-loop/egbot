import json, os, sys

txt = ""
for no in range(1,9):
    filename = "MathStackExchangeAPI_Part_" + str(no) + ".json"
    filepath = os.path.abspath("../../data/build/slim/"+filename)
    print("Opening " + filename)

    with open(filepath) as f:
        data = json.load(f)
        print("Finished loading")

        for i in range(0, len(data)):
            if("answers" in data[i].keys()):
                answers = []
                for j in range(0, len(data[i]["answers"])):
                    ans = dict()
                    ans["is_accepted"] = data[i]["answers"][j]["is_accepted"]
                    if (ans["is_accepted"]):
                        print("Found one " + str(len(txt)))
                        question = data[i]["body_markdown"]
                        answer = data[i]["answers"][j]["body_markdown"]
                        txt += question + " " + answer + " "
print("Saving ...")
with open("../../data/build/MathStackExchangeAPI.txt", "w") as outfile:
    outfile.write(txt)

print("Done :)")    