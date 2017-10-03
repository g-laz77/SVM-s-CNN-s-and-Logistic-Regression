import sys
from sklearn.linear_model import Lasso
import pandas as pd

train_file = sys.argv[1]
test_file = sys.argv[2]

datafile = pd.read_csv(train_file,header=None)
train_data = datafile.iloc[:,:11].values
train_class = datafile.iloc[:,11].values
# print train_data, train_class

model = Lasso(alpha=0.01, normalize=False, max_iter=1e5)
model.fit(train_data,train_class)

datafile = pd.read_csv(test_file,header=None)
test_data = datafile.iloc[:,:11].values
test_class = datafile.iloc[:,11].values
# print test_data
predicted_classes = model.predict(test_data)
count = 0
for i in range(len(predicted_classes)):
    if predicted_classes[i] >= 0.5 :
        predicted_classes[i] = 1
    elif predicted_classes[i] < 0.5 :
        predicted_classes[i] = 0
    
print predicted_classes
# print float(count)/test_class.shape[0]
score = accuracy_score(test_class, predicted_classes)
print(score)
