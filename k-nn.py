import csv
import random
import math
import operator
import numpy as np
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import stopwords

model = KeyedVectors.load('model.bin')

#Loading our dataset
def dataload(file,split,train=[],test=[]):
	#print(file)
	with open(file,'r') as csvfile:
		lines=csv.reader(csvfile)
		data=list(lines)
	#data = data.dropna(axis=1, how='any')
	for x in range(1, len(data)-1):
		#print(data[x])
		l=data[x]
		d = []
		for i in l:
			if i != '':
				d.append(i)
		#print(d)
		#break
		if(x<split):
			train.append(d)
		else:
			test.append(d)

def cosineDistance(x, y, length):
	#print(length)
	#print(len(y))
	distance=0
	magnitude_x=0
	magnitude_y=0
	#print(x[0])
	#print(y)
	product=0
	for i in range(0,len(x)):
		magnitude_x += x[i]*x[i]
	for i in range(0,len(y)):
		magnitude_y +=y[i]*y[i]	
	for i in range(0,len(x)):
		product+=x[i]*y[i]
	#print(product)	
	distance=float(product)/float((math.sqrt(magnitude_x)*math.sqrt(magnitude_y)))
	#print(distance)
	return(1-distance)
	

def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	a = np.asarray(a,dtype=float)
	b = np.asarray(b,dtype=float)
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	cos = dot_product / (norm_a * norm_b)
	return (1-cos)
	
def average_vec(sentence):
    words = sentence.split()
    word_vecs = [model[w] for w in words]
    #print(len(word_vecs))    
    #return word_vecs
    return (np.array(word_vecs).sum(axis=0)/len(word_vecs)).reshape(1,-1)
    
def cleanup(sentence):
	trainList=[]
	sentence=str(sentence)
	print(type(sentence))
	train_str=sentence.lower()
	words=words=nltk.word_tokenize(train_str)
	for word in words:
		if (word.isalpha() and not word in stopword_set):
			trainList.append(word)
	train_str=' '.join(trainList)
	return train_str	

def fullDistFinder(train,test):
	train1 = train[0:len(train)-1]
	train_num = []
	train_str = []
	test_num = []
	test_str = []

	print(len(test))	
	print('**********************************************')	
	for i in range(0,len(train1)):
		#print(train1[i])
		#print(test[i])	
		if train1[i].isdigit():
			train_num.append(float(train1[i]))
			test_num.append(float(test[i]))
		else:
			print(train1[i])
			x=cleanup(train1[i])
			train_str.append(x)	
			x=cleanup(test[i])
			test_str.append(x)
			
	dist1 = cosineDistance(train_num, test_num, len(train_num))
	trainS = ' '.join(train_str)
	testS = ' '.join(test_str)
	print(trainS)
	print(testS)
	print('**********************************************')	
	dist2 = cosineDistance(average_vec(trainS)[0], average_vec(testS)[0], len(train_str))
	#print(dist2)
	#return(0)
	dist = (dist1 + dist2)/2
	return dist


#Returning K nearest neighbours based on distance
def Neighbors(train, test, k):
	distances = []
	length = len(test)-1
	#print(train[0])
	#print(test)
	for x in range(len(train)):
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')	
		dist = fullDistFinder(train[x],test)
		print(dist)
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
		distances.append((train[x], dist))
	print(len(distances))	
	distances.sort(key=operator.itemgetter(1))
	print("----------------------")
	#print(distances)
	print("----------------------")	
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

#Voting the classes obtained from Neighbours
def Response(neighbors):
	Votes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in Votes:
			Votes[response] += 1
		else:
			Votes[response] = 1
	sortedVotes = sorted(Votes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

#Calculating Accuracy
def Accuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

#Combining all functions to get KNN
trainingSet=[]
test=[]
split = 3000
dataload('New.csv', split, trainingSet, test)
#print(trainingSet)
#print(test)
stopword_set = set(stopwords.words('english'))
trainingSet=trainingSet[:][1:]
#test=test[:][1:]

print('Train set: ',trainingSet[0])
print('Test set: ',test[0])
print(len(trainingSet[0]))
print(len(test[0]))

"""k=6
#for k in range(1,21):
predictions=[]
for x in range(len(test)):
	print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
	neighbors = Neighbors(trainingSet, test[x], k)
	print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
	result = Response(neighbors)
	predictions.append(result)
	#print('Predicted=',result,', Actual=',test[x][-1])
#accuracy =Accuracy(test, predictions)
#print(accuracy)
print(predictions)"""

