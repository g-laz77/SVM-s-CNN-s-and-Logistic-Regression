import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import IPython.display
import pandas as pd
import sys
from numpy import genfromtxt
from io import StringIO
from random import randint
from sklearn import preprocessing as pp

from sklearn.linear_model import LogisticRegression

def accuracy(final_ans,test_res):
	co=0
	i=0
	for i in xrange(len(final_ans)):
		if final_ans[i]==0 and test_res[i]==0:
			co+=1
		elif final_ans[i]==1 and test_res[i]==1:
			co+=1
		i+=1
	return (1.0*co)/i

def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    filen = open("img.png","w+")
    filen.write(IPython.display.display(IPython.display.Image(data=f.getvalue())))

direct = sys.argv[1]
file1 = "notMNIST_train_data.csv"
file2 = "notMNIST_train_labels.csv"
file3 = "notMNIST_test_data.csv"
file4 = "notMNIST_test_labels.csv"
X_train = np.array(pd.read_csv(direct+file1,header=None))

y_train = np.array(pd.read_csv(direct+file2,header=None))
X_test = np.array(pd.read_csv(direct+file3,header=None))

y_test = np.array(pd.read_csv(direct+file4,header=None))
pp.StandardScaler().fit(X_train)
pp.StandardScaler().fit(X_test)

# is C the inverse of regularization parameter?
clf = LogisticRegression(penalty='l2', C=1000)
clf.fit(X_train, y_train)

test_ans = clf.predict(X_test)

final_ans = np.zeros(np.size(test_ans))

for i in xrange(len(test_ans)):
    if test_ans[i] >= 0.5:
        final_ans[i] = 1

accuracy = accuracy(final_ans,y_test)

print accuracy

weights = clf.coef_

im = weights.reshape((28, 28))
plt.imshow(im)
plt.show()


# n = randint(1, 1000)
# label = y_test[n]
# im = X_test[n, :].reshape((28, 28))
# print(label)
# showarray(im)

