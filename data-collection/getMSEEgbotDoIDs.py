###########################################################################
## Description: Script to get StackExchange q&a eg ids using their API	###
## Status: abandoned, not needed anymore								###
###########################################################################

# -*- coding: utf-8 -*-
# import required libraries
import requests, json, time, datetime, csv, sys
from pandas.io.json import json_normalize

# data collection iteration (batch api requests) < should probs turn this into a args script
iteration = 1

# define stackexchange api request properties
filter = '!6xDWPLKRA4Po3zDjD7Z(1*.MxpZFLB*l01cDgRGZDCNOl9YQ1.39SHKhMAp0jLcns8urn'
key = '0KV1Jx2FaDYYrrCnFFqUQw(('
site = 'math.stackexchange'
tag = 'examples-counterexamples+probability'
max = 1511621138 #time.time()
min = 0

if len(sys.argv) > 2:
	aid = sys.argv[2]
else:
	aid = '2140493'
request_link = 'http://api.stackexchange.com/2.2/answers/'+ aid + '/?key=' + key + '&order=desc&sort=creation' + '&tagged=' + tag + '&site=' + site + '&filter=' + filter
print "Looking at ", request_link

# make api request 
r = requests.get(request_link)
temp = json.loads(r.text)
while 'error_id' in temp.keys():
	time.sleep(5)
	r = requests.get(request_link)
	temp = json.loads(r.text)	
data = temp

print data

# search for relevant info in the api response and construct dataset$ git checkout -b iss53
master_count = 0
dup_count = 0
final = []
if "items" in data.keys():
	for question in data["items"]:
		master_count += 1
		print question["items"]

else:
	print "===========API-ERROR============="
	print data
	

# print script summary
print "---------------END---------------"
print "Questions looked at: ", master_count
print "Duplicates found: ", dup_count
# if question:
# 	print "From ", datetime.datetime.fromtimestamp(int(question["creation_date"])).strftime('%Y-%m-%d %H:%M:%S'), "to ", datetime.datetime.fromtimestamp(int(max)).strftime('%Y-%m-%d %H:%M:%S')

# # exporting dataset to csv
# try:
# 	json_normalize(final).to_csv('mse_%d_%d_%d.csv' % (iteration, min, max), encoding='utf-8')
# except UnicodeEncodeError as e: print(e)
