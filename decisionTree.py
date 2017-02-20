import numpy as np
import math
import csv
import sys
import copy 
import random

class Node:
    def __init__(self, val):
        self.l = None
        self.r = None
        self.inter = True
        self.cla = False
        self.v = val
        self.data=None
        self.flag =None
        self.alive = True

    def print_tree(self):
    	if self.inter:
    		print (self.v);
    	else:
    		print (self.cla);
    	if self.l:
    		self.l.print_tree()
    	if self.r:
            self.r.print_tree()

    def __nodeCount(self,counter):#nodeCount
    	if self:
    		counter[0] += 1;
    	if self.l:
    		self.l.__nodeCount(counter)
    	if self.r:
            self.r.__nodeCount(counter)

    def nodeCount(self):
    	counter =np.zeros(1);
    	self.__nodeCount(counter);
    	return counter[0].astype('int');
    
    def __leaveCount(self, counter):
    	if self.inter == False:
    		counter[0] += 1;
    	if self.l:
    		self.l.__leaveCount(counter)
    	if self.r:
            self.r.__leaveCount(counter)

    def leaveCount(self):
    	counter =np.zeros(1);
    	self.__leaveCount(counter);
    	return counter[0].astype('int');

    def killNode(self,idx,step):
    	if self.inter:
    		if idx == step[0]:
    			self.alive = False;
    			step[0] += 1;
    		else:
    			step[0] += 1;
    			self.l.killNode(idx,step);
    			self.r.killNode(idx,step);
    def update(self):
    	if self.alive == False:
    		self.inter = False;
    		self.cla = mainClass(self.data);
    		self.l = None;
    		self.r = None;
    	if self.l:
    		self.l.update()
    	if self.r:
            self.r.update()

def entropy(dataVec):
	numInstance = dataVec.shape[0];
	numTrue = 0;
	numFalse = 0;

	for val in dataVec:
		if val:
			numTrue += 1;
		else:
   			numFalse += 1;
	possTrue = numTrue/numInstance;
	possFalse = numFalse/numInstance;
	log2True = 0;
	log2False = 0;

	if numTrue == 0:
		log2True =0;
	else:
		log2True = math.log(possTrue,2);

	if numFalse == 0:
		log2False =0;
	else:
		log2False = math.log(possFalse,2);

	ent = math.fabs((possTrue * log2True+ possFalse * log2False)); 
	return (ent);

def weightEntropy(attrVec, classVec):
	numInstance = attrVec.shape[0];
	numInTrue = 0;
	numInFalse = 0;
	VecInTrue = [];
	VecInFalse = [];
	idx=0;
	for val in attrVec:
		if val:
			VecInTrue = np.append(VecInTrue,classVec[idx]);
			numInTrue += 1;
		else:
   			VecInFalse = np.append(VecInFalse,classVec[idx]);
   			numInFalse+= 1;
		idx +=1;
	if numInTrue == 0:
		return numInFalse/numInstance*entropy(VecInFalse);
	elif numInFalse == 0:
		return numInTrue/numInstance*entropy(VecInTrue);
	else:
		return numInTrue/numInstance*entropy(VecInTrue)+numInFalse/numInstance*entropy(VecInFalse);

def findLowestEntropy(data, flag):
	minEnt = entropy(data[:,data.shape[1]-1]);
	rowClass = data.shape[1]-1;
	attrMat = data[:,0:rowClass];
	classVec = data[:,rowClass];
	# minEnt = oldEnt;
	minEntAttrIdx=-1;
	for idx in range(attrMat.shape[1]):
		if flag[idx]:
			entTemp = weightEntropy(attrMat[:,idx],classVec);
			if entTemp < minEnt:
				minEnt = entTemp;
				minEntAttrIdx = idx;
	flag[minEntAttrIdx]=False;
	return minEntAttrIdx;

def seperateData(data,idxAttr):
	attrVec=data[:,idxAttr];
	idx=0;
	idxTrue=np.zeros(data.shape[0], dtype=bool);
	idxFalse=np.zeros(data.shape[0], dtype=bool);
	for val in attrVec:
		if val:
			idxTrue[idx] = True;
		else:
			idxFalse[idx] = True;
		idx +=1;
	dataTrue = data[idxTrue,:];
	dataFalse = data[idxFalse,:];
	return dataTrue, dataFalse;

def mainClass(data):
	trueNum = 0;
	falseNum = 0;
	for val in data[:,data.shape[1]-1]:
		if val:
			trueNum +=1;
		else:
			falseNum +=1;

	if trueNum >= falseNum:
		return True;
	else:
		return False;

def builtDecisionTree(data,flag,node):
	decAttr = findLowestEntropy(data,flag);
	if decAttr ==-1:
		node.inter = False;
		node.cla = mainClass(data);
		return;

	node.v = decAttr;
	node.data = data;
	node.r = Node(-1);
	node.l = Node(-2);
	flag[decAttr] = False;
	dataTrue, dataFalse = seperateData(data,decAttr);
	node.flag = flag;
	# print(decAttr);
	flagFalse = copy.copy(flag);
	flagTrue = copy.copy(flag);


	if dataFalse.shape[0]!=0 and entropy(dataFalse[:,dataFalse.shape[1]-1]) != 0:
		builtDecisionTree(dataFalse, flagFalse, node.l);
	else:
		node.l.inter = False;
		node.l.cla = mainClass(dataFalse);
		# print (decAttr,dataFalse.shape[0], entropy(dataFalse[:,dataFalse.shape[1]-1]), dataFalse[:,dataFalse.shape[1]-1]);

	if dataTrue.shape[0]!=0 and entropy(dataTrue[:,dataTrue.shape[1]-1]) != 0:
		builtDecisionTree(dataTrue, flagTrue, node.r);
	else:
		node.r.inter = False;
		node.r.cla = mainClass(dataTrue);
		# print (decAttr,dataTrue.shape[0], entropy(dataTrue[:,dataTrue.shape[1]-1]), dataTrue[:,dataTrue.shape[1]-1]);

def useDecisionTree(data,root):
	predict = np.zeros(data.shape[0]);
	actual = data[:,data.shape[1]-1];
	for idx,row in enumerate(data):
		predictTemp = classify(row,root);
		predict[idx] = predictTemp;

	predict = predict.astype('bool_');
	return sum(predict == actual)/data.shape[0];

def classify(row,root):
	cla = False;
	if root.inter:
		if row[root.v]:
			return classify(row,root.r);
		else:
			return classify(row,root.l);
	else:
		return (root.cla);

def pruneTree(root, pruneFactor,attrNum):
	pruneNode = int(pruneFactor*attrNum);
	arr = np.array([]);
	node = root;
	nodeNum = root.nodeCount();
	leaveNum = root.leaveCount();
	interNodeNum = nodeNum-leaveNum;
	pruneNodeIdx=random.sample(range(interNodeNum), pruneNode);
	for idx in range(pruneNode):
		global step;
		step=np.zeros(1);
		root.killNode(pruneNodeIdx[idx],step);
	root.update();

# read data
fileTraining = "/Users/jarvisjr/Documents/17Spring/MachineLearning/assignment2/data_sets1/training_set.csv";
fileValidate = "/Users/jarvisjr/Documents/17Spring/MachineLearning/assignment2/data_sets1/validation_set.csv";
fileTest = "/Users/jarvisjr/Documents/17Spring/MachineLearning/assignment2/data_sets1/test_set.csv";
pruneFactor = 0.2;
# fileTraining=sys.argv[1];
# fileTest=sys.argv[2];
# pruneFactor = sys.argv[3];

	# read training data
raw_data = open(fileTraining)
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE);
Temp = np.array(list(reader))
header = Temp[0,];
dataTraining = Temp[1:,].astype('bool_');

	# read validation data
raw_data = open(fileValidate)
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE);
Temp = np.array(list(reader))
dataValidation = Temp[1:,].astype('bool_');

	# read test data
raw_data = open(fileTest)
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE);
Temp = np.array(list(reader))
dataTest = Temp[1:,].astype('bool_');

# setup basic parameters
numAttr = header.shape[0]-1;
numTrain = dataTraining.shape[0];
numTest = dataTest.shape[0];
numTest = dataTest.shape[0];
flag = np.ones(numAttr, dtype=bool);

# training
root = Node(-1);
builtDecisionTree(dataTraining,flag,root);
nodeNum = root.nodeCount();
leaveNum = root.leaveCount();
acTrain = useDecisionTree(dataTraining,root);

# pruning 

# testing
acTest = useDecisionTree(dataTest,root);
pruneTree(root,pruneFactor,numAttr);
acTest2 = useDecisionTree(dataTest,root);

print ("Number of training instances = ",numTrain)
print ("Number of training attributes=", numAttr)
print ("Total number of nodes in the tree =", nodeNum)
print ("Number of leaf nodes in the tree =",leaveNum)
print ("Accuracy of the model on the training dataset = ", acTrain)

print ();

print ("Number of test instances = ",numTest)
print ("Number of test attributes=", numAttr)
print ("Total number of nodes in the tree =", nodeNum)
print ("Number of leaf nodes in the tree =", leaveNum)
print ("Accuracy of the model on the test dataset = ", acTest)
print ("Accuracy 2 of the model on the test dataset = ", acTest2)
