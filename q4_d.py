import sys
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd

train_file = sys.argv[1]
test_file = sys.argv[2]

datafile = pd.read_csv(train_file,header=None)
train_data = datafile.iloc[:,:11].values
train_class = datafile.iloc[:,11].values
# print train_data, train_class

model = LinearRegression( normalize=True)
model.fit(train_data,train_class)

datafile = pd.read_csv(test_file,header=None)
test_data = datafile.iloc[:,:11].values
# print test_data
y = model.predict(test_data)
print y
