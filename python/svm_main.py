from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#input_file = open("data_clean_imputed.csv","r")
input_file = open("pca.csv","r")

lines = input_file.readlines()

TRAINING_SIZE = 316
CLASSIFICATION_TYPE = 2
PCA_NUM = 40;

X = []
y = []

test_X = []
test_y = []

count = 0
for line in lines:
	tokens = line.strip().split(",")
	if count < TRAINING_SIZE:
		X.append(map(float, tokens[0:PCA_NUM]))
		if CLASSIFICATION_TYPE == 2:
			y.append(int(tokens[len(tokens)-1]))
		elif int(tokens[len(tokens)-1]) == 1:
			y.append(0)
		else:
			y.append(1)
		count += 1
	else:
		test_X.append(map(float, tokens[0:PCA_NUM]))
		if CLASSIFICATION_TYPE == 2:
			test_y.append(int(tokens[len(tokens)-1]))
		elif int(tokens[len(tokens)-1]) == 1:
			test_y.append(0)
		else:
			test_y.append(1)
#print "Y\n", y
#print "TEST Y\n", test_y

#print "X: ", len(X), "\tY: ",len(y),"\tXTEST: ",len(test_X),"\tYTEST: ",len(test_y)

train_predictions = OneVsRestClassifier(LinearSVC()).fit(X, y).predict(X)
test_predictions = OneVsRestClassifier(LinearSVC()).fit(X, y).predict(test_X)

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
print confusion_matrix(y, train_predictions)
print "Classification Report (Train)"
print classification_report(y, train_predictions)

print "TEST RESULTS"
print "Test Accuracy: " + str(1-test_error)
print "Confusion Matrix (Test)"
print confusion_matrix(test_y, test_predictions)
print "Classification Report (Test)"
print classification_report(test_y, test_predictions)
