import json, os

print("EgBot Descriptive Statistics \n")

noOfQs = 0
noOfAs = 0
noOfAPerQ = 0
noOfQsWithA = 0
noOfQsWithAccA = 0

avgScorePerQ = 0
avgScorePerA = 0
avgScorePerAccA = 0
noOfQsWithPosScore = 0
noOfQsWithGoodScore = 0

lenOfQs = 0
lenOfAs = 0

for no in range(1,9):
    filename = "MathStackExchangeAPI_Part_" + str(no) + ".json"
    filepath = os.path.abspath("../../data/build/"+filename)
    print("Opening " + filename)
    with open(filepath) as f:
        data = json.load(f)
        print("Finished loading")

        slim = []
        for i in range(0, len(data)):
            noOfQs += 1
            lenOfQs += len(data[i]["body_markdown"])
            score = data[i]["score"]
            avgScorePerQ += score
            if (score >= 0):
                noOfQsWithPosScore += 1
            if (score > 0):
                noOfQsWithGoodScore += 1
            if("answers" in data[i].keys()):
                noOfQsWithA += 1
                for j in range(0, len(data[i]["answers"])):
                    noOfAs += 1
                    ans = dict()
                    ans["body_markdown"] = data[i]["answers"][j]["body_markdown"]
                    ans["is_accepted"] = data[i]["answers"][j]["is_accepted"]
                    ans["score"] = data[i]["answers"][j]["score"]
                    avgScorePerA += ans["score"]
                    lenOfAs += len(ans["body_markdown"])
                    if(ans["is_accepted"]):
                        noOfQsWithAccA += 1
                        avgScorePerAccA += ans["score"]
print()            
print("Total no of questions: ", noOfQs)
print("Total no of answers: ", noOfAs)

print()            
print("No of answers per question: ", noOfAs/noOfQs)
print("No of questions with at least one answer: ", noOfQsWithA)
print("No of questions with an accepted answer: ", noOfQsWithAccA)

print()            
print("Average question score: ", avgScorePerQ/noOfQs)
print("No of questions with a positive score: ", noOfQsWithPosScore)
print("No of questions with a good score: ", noOfQsWithGoodScore)
print("Average answer score: ", avgScorePerA/noOfAs)
print("Average accepted answer score: ", avgScorePerAccA/noOfQsWithAccA)
print()      


print()            
print("Average question length: ", lenOfQs/noOfQs)
print("Average answer length: ", lenOfAs/noOfAs)
print()          

print("Done :)")