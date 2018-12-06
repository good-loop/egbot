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

noOfAsWithPosScore = 0
noOfAsWithGoodScore = 0

noOfQsWithGoodScoreANoAccA = 0
noOfAsWithPosScoreNotAcc = 0
noOfAsWithGoodScoreNotAcc = 0

lenOfQs = 0
lenOfAs = 0

noOfClosed = 0
noOfClosedDuplicates = 0
noOfClosedOffTopic = 0
noOfClosedOpinion = 0
noOfClosedUnclear = 0

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

            if("closed_details" in data[i].keys()):
                noOfClosed += 1
                reason = data[i]["closed_details"]["reason"]
                if "duplicate" in reason:  noOfClosedDuplicates += 1;
                elif "off-topic" in reason:  noOfClosedOffTopic += 1;
                elif "opinion" in reason:  noOfClosedOpinion += 1;
                elif "unclear" in reason:  noOfClosedUnclear += 1;

            hasPosA = False
            hasGoodA = False
            hasAccA = False

            if("answers" in data[i].keys()):
                noOfQsWithA += 1
                for j in range(0, len(data[i]["answers"])):
                    noOfAs += 1
                    ans = dict()
                    ans["body_markdown"] = data[i]["answers"][j]["body_markdown"]
                    lenOfAs += len(ans["body_markdown"])

                    ans["score"] = data[i]["answers"][j]["score"]
                    avgScorePerA += ans["score"]
                    if (ans["score"] >= 0):
                        noOfAsWithPosScore += 1
                        hasPosA = True
                    if (ans["score"] > 0):
                        noOfAsWithGoodScore += 1
                        hasGoodA = True

                    ans["is_accepted"] = data[i]["answers"][j]["is_accepted"]
                    if(ans["is_accepted"]):
                        noOfQsWithAccA += 1
                        avgScorePerAccA += ans["score"]
                        hasAccA = True
                    else:
                        if (ans["score"] >= 0):
                            noOfAsWithPosScoreNotAcc += 1
                        if (ans["score"] > 0):
                            noOfAsWithGoodScoreNotAcc += 1
            if (not hasAccA):
                if (hasGoodA):
                    noOfQsWithGoodScoreANoAccA += 1
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

print()
print("Average answer score: ", avgScorePerA/noOfAs)
print("Average accepted answer score: ", avgScorePerAccA/noOfQsWithAccA)
print("No of answers with a positive score: ", noOfAsWithPosScore)
print("No of answers with a good score: ", noOfAsWithGoodScore)

print()
print("No of questions with a good score answer but no accepted answer: ", noOfQsWithGoodScoreANoAccA)
print("No of answers with a positive score but not accepted: ", noOfAsWithPosScoreNotAcc)
print("No of answers with a good score but not accepted: ", noOfAsWithGoodScoreNotAcc)

print()            
print("Average question length: ", lenOfQs/noOfQs)
print("Average answer length: ", lenOfAs/noOfAs)

print()            
print("Number of closed questions: ", noOfClosed)
print("Number of duplicate questions: ", noOfClosedDuplicates)
print("Number of off-topic questions: ", noOfClosedOffTopic)
print("Number of opinion-based questions: ", noOfClosedOpinion)
print("Number of unclear questions: ", noOfClosedUnclear)

print()      
print("Done :)")