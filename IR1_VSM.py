import glob
import pandas as pd
import numpy as np
import os
import math
import operator
from sklearn.metrics.pairwise import cosine_similarity
import time
from progressbar import *

TF_arrD = np.zeros((51249,2265),float) 
#取Doc跟Qu最大的index值為字典總字數,優:能快速建立(不用count),缺:很多項值是0(2265篇Doc皆無此值)

list1 = glob.glob(r"Document/*")
print("total Document = ",len(list1)) 


widgets = ['Loading Document: ',Percentage(), ' ', Bar('#'),' ', Timer()]
progress = ProgressBar(widgets=widgets)
for x in progress(range(len(list1))):
	f = open(list1[x], 'r')
	line1 = f.readline()
	line2 = f.readline()
	line3 = f.readline()

	A = f.read()
	if A =='': break
	A = A.replace("-1","")
	A = A.replace("-n","")
	doc = A.split()
	for i in range(len(doc)):
		TF_arrD[int(doc[i]),x] += 1
f.close()

#--------------------------------------------------

TF_arrQ = np.zeros((51249,16),float)
#取Doc跟Qu最大的index值為字典總字數,優:能快速建立(不用count),缺:很多項值是0(16個Qu皆無此值)
list2 = glob.glob(r"Query/*")
print("total Query = ",len(list2))

widgets = ['Loading Query: ',Percentage(), ' ', Bar('#'),' ', Timer()]
progress = ProgressBar(widgets=widgets)
for y in progress(range(len(list2))):
	f = open(list2[y], 'r')
	B = f.read()
	if B =='': break
	B = B.replace("-1" , "") 
	B = B.replace("\n" , "")
	qu = B.split()
	for i in range(len(qu)):
		TF_arrQ[int(qu[i]),y] += 1
f.close()

#--------------------------------------------------

IDF_arr = np.zeros(51249,float)

widgets = ['Doing IDF: ',Percentage(), ' ', Bar('#'),' ', Timer()]
progress = ProgressBar(widgets=widgets)
for m in progress(range(51249)):
	for n in range(len(list1)):
		if TF_arrD[m,n] != 0:
			IDF_arr[m]+=1


for m in range(51249):
	if IDF_arr[m] == 0:
		IDF_arr[m] = 0
	else:
		IDF_arr[m] = math.log(((len(list1))/IDF_arr[m])) #!!!!!!改權重

#--------------------------------------------------
tfidfD = np.zeros((51249,len(list1)),float)
print("start Document TF-IDF table")

widgets = ['Doing Document TF-IDF: ',Percentage(), ' ', Bar('#'),' ', Timer()]
progress = ProgressBar(widgets=widgets)
for i in progress(range(len(list1))):
	for j in range(51249):
		tfidfD[j,i] = TF_arrD[j,i] * IDF_arr[j] #!!!!!!改權重
f.close()

#--------------------------------------------------

tfidfQ = np.zeros((51249,len(list2)),float)
print("start Query TF-IDFT table")

widgets = ['Doing Query TF-IDF: ',Percentage(), ' ', Bar('#'),' ', Timer()]
progress = ProgressBar(widgets=widgets)
for i in progress(range(len(list2))):
	for j in range(51249):
		tfidfQ[j,i] = TF_arrQ[j,i] * IDF_arr[j] #!!!!!!改權重


#--------------------------------------------------

#D_tfidf,Q_tfidf
PointTable = np.zeros((len(list2),len(list1)),float)
# print(tfidfD.shape)
# print(NewtfidfQ.shape)

widgets = ['caculating cosine_similarity: ',Percentage(), ' ', Bar('#'),' ', Timer()]
progress = ProgressBar(widgets=widgets)
for x in progress(range(len(list2))): #Q有16個
	for y in range(len(list1)): #D有2265個
		PointTable[x,y] = cosine_similarity(tfidfD[:,y].reshape(1, -1),tfidfQ[:,x].reshape(1, -1))

print(PointTable)

result = open("Ranking.txt", 'w')
result.write("Query,RetrievedDocuments\n")

for i in range(len(list2)):
	D = {}
	result.write(list2[i])
	for j in range(len(list1)):
		D.update({list1[j]:PointTable[i,j]})
	D = sorted(D.items(),key=operator.itemgetter(1),reverse = True)
	for k in range(len(D)):
	 result.write(D[k][0])

