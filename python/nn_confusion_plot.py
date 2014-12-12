from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

input_arr_test = [[75,6,2,2,1,0,0,0,1,0,0,0,0],[2,4,0,1,1,0,0,1,1,1,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0],[0,1,0,3,1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,6,0,0,0,1,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,2,2,1,0,1,9,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,4]]
input_arr_train = [[167,0,0,0,0,0,0,0,0,0,0,0,0],[0,33,0,0,0,0,0,0,0,0,0,0,0],[0,0,12,0,0,0,0,0,0,0,0,0,0],[0,0,0,9,0,0,0,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0,0,0,0,0],[0,0,0,0,0,17,0,0,0,0,0,0,0],[0,0,0,0,0,0,2,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,5,0,0,0,0],[0,0,0,0,0,0,0,0,0,39,0,0,0],[0,0,0,0,0,0,0,0,0,0,3,0,0],[0,0,0,0,0,0,0,0,0,0,0,4,0],[0,0,0,0,0,0,0,0,0,0,0,0,16]]

conf_arr = np.array(input_arr_train, np.int32)
filename = "nn_train"

font = {'size'   : 17}
plt.rc('font', **font)

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