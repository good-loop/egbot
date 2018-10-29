## neat little converter that takes egbot data from ES and outputs q&a pairs in a txt file
## status: work in progress

# TODO: figure out format of output data
from elasticsearch import Elasticsearch
import os

print("\nHi, I'm just starting up")

# where to save results
saved_file = "egbot"
print("Gonna save things in " + os.path.abspath(saved_file))

# new data structure for indexing into es
#qa_pairs = dict()

# connecting to elastic search, expected to be on localhost port 9200
es = Elasticsearch()
print("\nConnected to ElasticSearch")

print("\nGetting all the EgBot records...")
#res = es.search(index="egbot.sequestion_aug18", doc_type="", body={"query": {"match_all": {}}})
#print("%d documents found" % res['hits']['total'])

page = es.search(index="egbot.sequestion_aug18", doc_type="", scroll='2m', 
		size = 1000, body={"query": {"match_all": {}}})
sid = page['_scroll_id']
scroll_size = page['hits']['total']
#print("%d documents found" % res['hits']['total'])

div = 1 #224357
count = 0 # number of q&a pairs, where the question was answered and the answer was accepted by the asker
no_of_seen_pages = 0 # scrolling position
# Start scrolling
while (scroll_size > 0):
	print "\nScrolling..."
	page = es.scroll(scroll_id = sid, scroll = '2m')

	# Update the scroll ID
	sid = page['_scroll_id']

	# Get the number of results that we returned in the last scroll
	scroll_size = len(page['hits']['hits'])
	no_of_seen_pages += scroll_size
	print "scroll size: " + str(no_of_seen_pages)
	#print(len(res['hits']['hits']))

	for eg in page['hits']['hits']:
		src  = eg['_source']
		if 'accepted_answer_id' in src.keys() and src['is_answered']:
			count += 1
			qbod = ''.join(src['body_markdown']).encode('utf-8')
			for ans in src['answers']:
				if ans['is_accepted']:
					if src['accepted_answer_id'] == ans['answer_id']:
						abod = ''.join(ans['body_markdown']).encode('utf-8')
			#qa_pairs[count]['question'] = qbod
			#qa_pairs[count]['answer'] = abod

			# writing results to file
			with open(saved_file+'_'+str(div)+'.txt', "a") as f:
				#print "---------------------------------------------"
				#print "Q:"+qbod+"\n\nA:"+abod
				f.write(qbod+' '+abod+' ')
	div = int(count/600) + 1
	print '\ndiv:' + str(div)
	print "count: " + str(count)

print("\nI'm done! Saved it all in " + os.path.abspath(saved_file))
print("\nEnjoy ^.^\n")

# Let's see some examples
#egbot = res['hits']['hits'][0]['_source']
#eg_qbod = egbot['body_markdown']
#eg_aid = egbot['answers'][0]['answer_id']
#eg_abod = egbot['answers'][0]['body_markdown']
#print('\nExample Q&A pair:\n--------------------------------------------\nQ: ' 
#	+ eg_qbod 
#	+ '\n------------------------------------------------------\nA: ' 
#	+ eg_abod)









