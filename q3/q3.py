import matplotlib
import numpy as np
import PIL.Image
import IPython.display

from numpy import genfromtxt
from cStringIO import StringIO
from random import randint
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


X_train = genfromtxt('notMNIST_train_data.csv', delimiter=',')
X_train = preprocessing.scale(X_train)
# X_train = preprocessing.normalize(X_train, norm='l2')
y_train = genfromtxt('notMNIST_train_labels.csv', delimiter=',')
X_test = genfromtxt('notMNIST_test_data.csv', delimiter=',')
X_test = preprocessing.scale(X_test)
# X_test = preprocessing.normalize(X_test, norm='l2')
y_test = genfromtxt('notMNIST_test_labels.csv', delimiter=',')

# is C the inverse of regularization parameter?
clf = LogisticRegression(penalty='l1', C=0.01)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print score


n = randint(1, 1000)
label = y_test[n]
im = X_test[n, :].reshape((28, 28))
print(label)
showarray(im)

