# Decision Tree

The ID3 decision tree learning algorithm is implemented to build a binary decision tree classifier using python. The model is builded using the training dataset, check the model and prune it with the validation dataset, and test it using the testing dataset.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

###Data Format

-The program read four arguments from the command line – complete path of the training dataset, complete path of the validation dataset, complete path of the test dataset, and the pruning factor.

-The datasets can contain any number of Boolean attributes and one Boolean class label. The
class label will always be the last column.

-The first row will define column names and every subsequent non-blank line will contain a
data instance. If there is a blank line, your program should skip it

-The example of datasets will be provided in [data_sets1](data_sets1)

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
python3 decisionTree.py _complete path of the training dataset_ _complete path of the validation dataset_ _complete path of the test dataset_ _prune factor (optional)_
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment
import numpy as np
import math
import csv
import sys
import copy 
import random


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
