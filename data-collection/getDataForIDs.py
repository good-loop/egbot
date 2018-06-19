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

# data collection iteration (batch api requests) < should probs turn this into a args script
iteration = 1

# define stackexchange api request properties
filter = '!asyat-glvTDN7Q*KS8FC0X2Ds8E427nbJlZsxliDugZwP6._EWQ0H)6SoS2c'
key = 'EuiPGzIIW4aHrRAJq5JQYA(('
site = 'math.stackexchange'
#tag = 'examples-counterexamples+probability'
max = time.time()
min = 0

with open('input.txt', 'r') as f:
	read_data = f.readlines()

# split ids into batches of 100 to respect api limit
qids = ""
if len(read_data) >= 100:
	for qid in read_data[:99]:
		qids += ";".join(qid.split("\n"))
else:
	for qid in read_data:
		qids += ";".join(qid.split("\n"))

request_link = 'http://api.stackexchange.com/2.2/questions/' + qids + '/?key=' + key + '&order=desc&sort=creation' + '&site=' + site + '&filter=' + filter
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

# search for relevant info in the api response and construct dataset
master_count = 0
query_counter = 1
req_count = 1
dup_count = 0
counter = 60
final = []
current_max = max
tryagain = False
it = 1
while data:
	if "items" in data.keys():
		if len(data["items"]) > 0:
			for question in data["items"]:
				master_count += 1
				if int(question["answer_count"]) > 0:
					answers = question["answers"]
					listOfAnswers = []
					for answer in answers:
						listOfAnswers.append(answer["body_markdown"])
					question["body_markdown_answers"] = listOfAnswers
					final.append(question)
			
			# exporting dataset to csv
			try:
				json_normalize(final).to_csv('data/ind/ind_mse_%d_%d_%d_%d.csv' % (iteration,it, int(question["creation_date"]), max), sep=',', encoding='utf-8')
				it += 1
			except UnicodeEncodeError as e: print(e)

			if len(read_data) > query_counter*100:			
				for qid in read_data[query_counter*100:query_counter*101-1]:
					qids += qid + ";"
				qids += read_data[query_counter*101]
				query_counter += 1

				# do another api request to get the next 100 posts
				print "--------------NEXT---------------"
				request_link = 'http://api.stackexchange.com/2.2/questions/' + qids + '/?key=' + key + '&order=desc&sort=creation&pagesize=100&min=' + str(int(min)) + '&max=' + str(question["creation_date"]) + '&site=' + site + '&filter=' + filter
				current_max = int(question["creation_date"])
				print 'Questions looked at: ', master_count, ", Requests made: ", req_count
				print "Looking at ", request_link
				try:
					req_count += 1
					r = requests.get(request_link)
					tryagain = False
				except requests.exceptions.RequestException as e: 
					tryagain = True
					print e
				temp = json.loads(r.text)
				while 'error_id' in temp.keys() or tryagain:
					print 'Questions looked at: ', master_count
					print 'Sleeping for ' + str(counter) + ' seconds ...'
					time.sleep(counter)
					counter = 3 * counter
					try:
						req_count += 1
						r = requests.get(request_link)	
						tryagain = False				
					except requests.exceptions.RequestException as e: 
						tryagain = True
						print e
					temp = json.loads(r.text)	
				data = temp	
				counter = 60
			else:
				break
		else:
			print "Error: Couldn't find questions in this response"
	else:
		print "===========API-ERROR============="
		print data
		break

# print script summary
print "---------------END---------------"
print "Questions looked at: ", master_count, ", Requests made: ", req_count
#print "Duplicates found: ", dup_count
if question:
	print "From ", datetime.datetime.fromtimestamp(int(question["creation_date"])).strftime('%Y-%m-%d %H:%M:%S'), "to ", datetime.datetime.fromtimestamp(int(max)).strftime('%Y-%m-%d %H:%M:%S')

# exporting dataset to csv
try:
	json_normalize(final).to_csv('data/ind/ind_mse_aggregate_%d_%d_%d.csv' % (iteration, int(question["creation_date"]), max), sep=',', encoding='utf-8')
except UnicodeEncodeError as e: print(e)
