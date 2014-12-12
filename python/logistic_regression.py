from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

def generateConfusionMatrixFigure(conf_arr, filename):
	norm_conf = []
	for i in conf_arr:
	    a = 0
	    tmp_arr = []
	    a = sum(i, 0)
	    for j in i:
	        tmp_arr.append(float(j)/float(a))
	    norm_conf.append(tmp_arr)

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
	                interpolation='nearest')

	width = len(conf_arr)
	height = len(conf_arr[0])

	for x in xrange(width):
	    for y in xrange(height):
	        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')

	cb = fig.colorbar(res)
	axis_labels = ['1','2','3','4','5','6','7','8','9','10','14','15','16']
	plt.xticks(range(width), axis_labels)
	plt.yticks(range(height), axis_labels)
	plt.savefig(filename + '.eps', format='eps')
	plt.savefig(filename + '.png', format='png')

input_file = open("../data/data_clean_imputed.csv","r")
#input_file = open("../data/pca.csv","r")

lines = input_file.readlines()

TRAINING_SIZE = 316
# 1 for binary, 2 for multiclass
CLASSIFICATION_TYPE = 2
NUM_PCA = 270;

X = []
y = []

test_X = []
test_y = []

count = 0
for line in lines:
	tokens = line.strip().split(",")
	if count < TRAINING_SIZE:
		X.append(map(float, tokens[0:NUM_PCA]))
		if CLASSIFICATION_TYPE == 2:
			y.append(int(tokens[len(tokens)-1]))
		elif int(tokens[len(tokens)-1]) == 1:
			y.append(0)
		else:
			y.append(1)
		count += 1
	else:
		test_X.append(map(float, tokens[0:NUM_PCA]))
		if CLASSIFICATION_TYPE == 2:
			test_y.append(int(tokens[len(tokens)-1]))
		elif int(tokens[len(tokens)-1]) == 1:
			test_y.append(0)
		else:
			test_y.append(1)

lrm = LogisticRegression() 
lrm.fit(X,y)

#clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
#clf.fit(X,y)

train_predictions = lrm.predict(X)
test_predictions = lrm.predict(test_X)

#train_predictions = OneVsRestClassifier(LinearSVC()).fit(X, y).predict(X)
#test_predictions = OneVsRestClassifier(LinearSVC()).fit(X, y).predict(test_X)

train_missed = 0
test_missed = 0
for i in range(0, len(train_predictions)):
	if train_predictions[i] != y[i]:
		train_missed += 1
for i in range(0, len(test_predictions)):
	if test_predictions[i] != test_y[i]:
		test_missed += 1

train_error = train_missed*1.0/len(train_predictions)
test_error = test_missed*1.0/len(test_predictions)

print "TRAIN RESULTS"
print "Train Accuracy: " + str(1-train_error)
print "Confusion Matrix (Train)"
conf_array_train = confusion_matrix(y, train_predictions)
generateConfusionMatrixFigure(conf_array_train, 'logistic_train')
print conf_array_train
print "Classification Report (Train)"
print classification_report(y, train_predictions)

print "TEST RESULTS"
print "Test Accuracy: " + str(1-test_error)
print "Confusion Matrix (Test)"
conf_arr_test = confusion_matrix(test_y, test_predictions)
generateConfusionMatrixFigure(conf_arr_test, 'logistic_test')
print conf_arr_test
print "Classification Report (Test)"
print classification_report(test_y, test_predictions)