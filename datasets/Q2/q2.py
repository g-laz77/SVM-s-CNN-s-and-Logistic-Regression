from keras.models import Sequential,Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Input,Dense, Dropout, Activation, Flatten
from matplotlib import pyplot as plt
import numpy as np
import cPickle
import tensorflow as tf
# fix random seed for reproducibility
batch_size = 32 
num_epochs = 20 
kernel_size = 3 
pool_size = 2 
conv_depth_1 = 32 
conv_depth_2 = 64 
drop_prob_1 = 0.25 
drop_prob_2 = 0.5 
hidden_size = 512 
np.random.seed(7)
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
 
def cnn(X_train,Y_train):  
    inp = Input(shape=(3, 32, 32))
    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
    conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size),dim_ordering="th")(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
    conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
    conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size),dim_ordering="th")(conv_4)
    drop_2 = Dropout(drop_prob_1)(pool_2)
    # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(10, activation='softmax')(drop_3)
    model = Model(inputs=inp, outputs=out) 
    model.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
    model.fit(X_train, Y_train,                
            batch_size=batch_size, epochs=num_epochs,
            verbose=1, validation_split=0.1) 
    # model.evaluate(X_test, Y_test, verbose=1)  

if "__name__" != "__main__":
    train_data = unpickle("data_batch_1")["data"]
    train_labels = unpickle("data_batch_1")["labels"]
    for i in range(2,6):
        a = unpickle("data_batch_"+str(i))
        train_data = np.append(train_data,a["data"],axis=0)
        train_labels += a["labels"]
    b = unpickle("batches.meta")
    label_names = b["label_names"]
    #reshaping the training data and their classes
    train_data = np.reshape(train_data,(train_data.shape[0], 3, 32, 32))
    train_labels = np_utils.to_categorical(train_labels, 10)

    train_data = train_data.astype('float32')
    train_data /= np.max(train_data)
    print train_data.shape
    print train_labels.shape
    print label_names
    cnn(train_data,train_labels)