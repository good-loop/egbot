###########################################################################
## Description: Script to get StackExchange q&a eg ids using their API	###
## Status: WIP															###
###########################################################################

# input txt 231231;3232;32323;322323  
# output csv 

# -*- coding: utf-8 -*-
# import required libraries
import requests, json, time, datetime, csv, sys
from pandas.io.json import json_normalize

# path to data folder
path = './data'

# define iteration run of this script for batch running
# data collection iteration (batch api requests) < should probs turn this into a args script
iteration = 1

# define stackexchange api request properties
filter = '!asyat-glvTDN7Q*KS8FC0X2Ds8E427nbJlZsxliDugZwP6._EWQ0H)6SoS2c'
key = 'EuiPGzIIW4aHrRAJq5JQYA(('
site = 'math.stackexchange'
#tag = 'examples-counterexamples+probability'
max = time.time()
min = 0

with open(path+"/input.json", "r") as read_file:
    whole = json.load(read_file)

qidList = []
aids = []
for line in whole:
    if line['qid'] not in qidList:
        qidList.append(str(line['qid']))
    aids.append(line['aid'])

qids = []
i = 0
while i < len(qidList)/100:
    qids.append(";".join(qidList[i*100:(i+1)*100]))
    i += 1
qids.append(";".join(qidList[i*100:len(qidList)]))

# todo: figure out manual extraction from generated answer body, should i mark index/ location or the actual text (maybe both for now?) 

request_link = 'http://api.stackexchange.com/2.2/questions/' + qids[0] + '/?key=' + key + '&pagesize=100' + '&site=' + site + '&filter=' + filter
print "Looking at ", request_link

# make api request 
r = requests.get(request_link)
temp = json.loads(r.text)
while 'error_id' in temp.keys():
	print "Sleeping for 60 seconds ..."
	time.sleep(60)
	r = requests.get(request_link)
	temp = json.loads(r.text)	
data = temp

#todo: right now it does a req for the spec qids and gets all the data as per filter, will need to be moded to have extra fields relating to spec aids

# search for relevant info in the api response and construct dataset
ans_count = 0
req_count = 1
counter = 60
final = []
current = max
tryagain = False
section = 0
q_count = 0
check = True
while check:
    if "items" in data.keys():
        if len(data["items"]) > 0:
            for question in data["items"]:
                q_count += 1
                for ans in question["answers"]:
                    if ans["answer_id"] in aids:
                        question_ext = question.copy()
                        question_ext["answers"] = ""
                        question_ext["body_markdown"] = ""
                        question_ext["egbot_answer_body"] = ""#ans["body_markdown"]
                        question_ext["egbot_answer_id"] = ans["answer_id"]
                        for line in whole:
                            if line["aid"] == ans["answer_id"]:
                                question_ext["egbot_answer_label"] = line["egbotdo"]
                                break
                        ans_count += 1
                        final.append(question_ext)
            
            current = question["creation_date"]
        else:
            print "Error: Couldn't find questions in this response"
    else:
        print "===========API-ERROR============="

    section += 1
    if section < len(qids):
        request_link = 'http://api.stackexchange.com/2.2/questions/' + qids[section] + '/?key=' + key + '&pagesize=100' + '&site=' + site + '&filter=' + filter
        print "Looking at ", request_link
        req_count += 1

        # make api request 
        r = requests.get(request_link)
        temp = json.loads(r.text)
        while 'error_id' in temp.keys():
            print "Sleeping for 60 seconds ..."
            time.sleep(60)
            r = requests.get(request_link)
            temp = json.loads(r.text)	
        data = temp
    else:
        check = None

# print script summary
print "---------------END---------------"
print "Questions looked at: ", q_count, "Answers looked at: ", ans_count, ", Requests made: ", req_count

# exporting dataset
with open(path + '/output_mse_agg.json', 'w') as outfile:  
    json.dump(final, outfile)
with open(path + '/output_mse_agg_es.txt', 'w') as f:
    for i in range(0,len(final)):
        print >> f, '{\"index\":{}}\n', final[i] 
try:
	json_normalize(final).to_csv(path + '/output_mse_aggregate_%d_%d_%d.csv' % (iteration, int(current), int(max)), sep=',', encoding='utf-8')
except UnicodeEncodeError as e: print(e)

