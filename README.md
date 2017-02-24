# Decision Tree

The ID3 decision tree learning algorithm is implemented to build a binary decision tree classifier using python. The model is builded using the training dataset, check the model and prune it with the validation dataset, and test it using the testing dataset.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

###Data Format

*The program read four arguments from the command line – complete path of the training dataset, complete path of the validation dataset, complete path of the test dataset, and the pruning factor.

*The datasets can contain any number of Boolean attributes and one Boolean class label. The
class label will always be the last column.

*The first row will define column names and every subsequent non-blank line will contain a
data instance. If there is a blank line, your program should skip it

*The example of datasets will be provided in [data_sets1](data_sets1)

### Prerequisites

-python3
-Numpy

The package, Numpy, can be installed with: 

```
pip3 install numpy
```

### Running

The progame will be excuted from the command line with:

```
python3 decisionTree.py _complete path of the training dataset_ _complete path of the validation dataset_ _complete path of the test dataset_ 
```

### Classification Report

After running the program, the report of the result will be generated as:

```
Random-generated Accuracy
-------------------------------------
Number of training instances =  600
Number of training attributes= 20
Total number of nodes in the tree = 1153
Number of leaf nodes in the tree = 577
Average depth of the tree = 10.742
Accuracy of the model on the training dataset =  1.0

Number of validation instances =  2000
Number of validation attributes= 20
Accuracy of the model on the validation dataset before pruning =  0.603

Number of test instances =  2000
Number of test attributes= 20
Accuracy of the model on the test dataset before pruning =  0.6035

Pre-Pruned Accuracy
-------------------------------------
Number of training instances =  600
Number of training attributes= 20
Total number of nodes in the tree = 275
Number of leaf nodes in the tree = 138
Average depth of the tree = 8.2266
Accuracy of the model on the training dataset =  1.0

Number of validation instances =  2000
Number of validation attributes= 20
Accuracy of the model on the validation dataset before pruning =  0.759

Number of test instances =  2000
Number of test attributes= 20
Accuracy of the model on the test dataset before pruning =  0.7585
```

## Deployment

### Packages
*numpy - N-dimensional array computing
*csv - .cvs file reading
*sys - arguments reading
*copy - data copying
*random - random number generating
*math - scientific computing

### Functions
'''
def entropy(dataVec):...
	#count the entropy of all instances in one selected attribute

def weightEntropy(attrVec, classVec):...
	#find the information gain

def findLowestEntropy(data, flag):...
	#find the attribute with lowest entropy

def seperateData(data,idxAttr):...
	#seperate data base on the value of the selected attribute

def mainClass(data):...
	#find the major class of data

def builtDecisionTree(data,flag,node):...
	#construct a decision tree by using ID3 algorithm 

def builtDecisionTreeRandom(data,flag,node):...
	#construct a decision tree by randomly selecting attributes

def findRandom(flag):...
	#find a attribute randomly

def useDecisionTree(data,root):...
	#classify the data

def classify(row,root):...
	#throw a instance into the decision tree to get its class

def pruneTree(root, pruneFactor,attrNum):...
	#select the node of the tree randomly and prune them

def printTree(root,header):...
	# print the decision tree
'''

## Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## Authors
* **Tzu-yi Kuo** - *Initial work* - [TYKuoTaiwan](https://github.com/TYKuoTaiwan)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
