###########################################################################
## Description: Script to get StackExchange q&a eg ids using their API	###
## Status: WIP															###
###########################################################################

# -*- coding: utf-8 -*-
# import required libraries
import requests, json, time, datetime, csv, sys
from pandas.io.json import json_normalize

# path to data folder
path = '/home/irina/data'
 
# data collection iteration (batch api requests) < should probs turn this into a args script
iteration = 8																				## UPDATE THIS: to improve data management

# define stackexchange api request properties	
filter = '!asyat-glvTDN7Q*KS8FC0X2Ds8E427nbJlZsxliDugZwP6._EWQ0H)6SoS2c' 
key = '564*QrqqIA5WwT1eXCHsTA(('														## UPDATE THIS: to improve speed
site = 'math.stackexchange'
#tag = 'examples-counterexamples+probability'
max = 1367752303 #int(time.time()) 															## UPDATE THIS: to improve data management
min = 0

request_link = 'http://api.stackexchange.com/2.2/questions/?key=' + key + '&order=desc&sort=creation&pagesize=100&min=' + str(int(min)) + '&max=' + str(max) + '&site=' + site + '&filter=' + filter
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
req_count = 1
dup_count = 0
counter = 60
final = []
current = max
tryagain = False
it = 1
while data:
	if "items" in data.keys():
		if len(data["items"]) > 0:
			for question in data["items"]:
				master_count += 1
				final.append(question)
			
			current = question["creation_date"]
			# exporting dataset to json
			with open(path + '/getAllData_V2_mse_%d_%d_%d_%d.json' % (iteration, it, int(current), int(max)), 'w') as outfile:  
				json.dump(final, outfile)
			it += 1

			# do another api request to get the next 100 posts
			print "--------------NEXT---------------"
			request_link = 'http://api.stackexchange.com/2.2/questions/?key=' + key + '&order=desc&sort=creation&pagesize=100&min=' + str(int(min)) + '&max=' + str(current) + '&site=' + site + '&filter=' + filter
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
			print "Error: Couldn't find questions in this response"
			print request_link
			print data
			break
	else:
		print "===========API-ERROR============="
		print data
		break

# print script summary
print "---------------END---------------"
print "Questions looked at: ", master_count, ", Requests made: ", req_count
#print "Duplicates found: ", dup_count
#if question:
#	print "From ", datetime.datetime.fromtimestamp(int(question["creation_date"])).strftime('%Y-%m-%d %H:%M:%S'), "to ", datetime.datetime.fromtimestamp(int(max)).strftime('%Y-%m-%d %H:%M:%S')

# exporting dataset to json
with open(path + '/getAllData_V2_mse_aggregate_%d_%d_%d.json' % (iteration, int(current), int(max)), 'w') as outfile:  
    json.dump(final, outfile)
