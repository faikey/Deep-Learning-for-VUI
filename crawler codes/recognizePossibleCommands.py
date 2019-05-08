import nltk
from nltk import Tree
from pycorenlp import StanfordCoreNLP

#nlp = StanfordCoreNLP('http://localhost:9000')

#first sparse words
text = "help ask me to play a podcast for example you can say play the dr. Laura program can I play a podcast for you"
text = nltk.word_tokenize(text)
print(text)
index = []
count = 0
for i in text:
	if(i == "ask" or i == "like" or i == "say"):
		index.append(count)
	count = count + 1
#print(index)

index_or = []
count = 0
for i in text:
	if(i == "or"):
		index_or.append(count)
	count = count + 1
#print(index_or)



subtexts = []
#subtexts.append(text[index[0]+1:index_or[0]])
#print(str(subtexts))
for j in range(len(index)):
	#print(j, subtexts)
	if(len(index_or)!=0):

		if(index[j]<index_or[0]):
			subtexts.append(" ".join(text[index[j]+1:index_or[0]]))
		if(index[j]>index_or[0] and index[j]<index_or[-1]):
			for i in range(len(index_or)):
				dis = index_or[i]-index[j]
				if(dis > 0):
					break
			subtexts.append(" ".join(text[index[j]+1:index[j]+dis]))
		elif(index[j]>index_or[-1]):
			subtexts.append(" ".join(text[index[j]+1:-1]))
	else:
		subtexts.append(" ".join(text[index[j]+1:-1]))

for k in range(len(index_or)):
	if(k+1 <= len(index_or) -1):
		subtexts.append(" ".join(text[index_or[k]+1:index_or[k+1]]))


for t in subtexts:
	print(type(t))

#output = nlp.annotate(text, properties={
        #'annotators': 'tokenize,ssplit,pos,depparse,parse',
        #'outputFormat': 'json'
    #})
#print(output['sentences'][0]['parse'])

#t = Tree.fromstring(str(output['sentences'][0]['parse']))
#t.draw()

#subtexts = []

#for subtree in t.subtrees():
	#if subtree.label() == "SBAR":
		#print (subtree.leaves())
		#subtexts.append(' '.join(subtree.leaves()))


#for i in reversed(range(len(subtexts)-1)):
    #subtexts[i] = subtexts[i][0:subtexts[i].index(subtexts[i])]
    #print(subtexts[i])

#for text in subtexts:
    #output = nlp.annotate(text, properties={
            #'annotators': 'tokenize,ssplit,pos,depparse,parse',
            #'outputFormat': 'json'
        #})
    #print(output['sentences'][0]['parse'])

    #t = Tree.fromstring(str(output['sentences'][0]['parse']))
 

