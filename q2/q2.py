from keras.models import Sequential,Model
from keras.utils import np_utils
from keras.layers import Input,Dense, Dropout, Activation, Flatten, Conv2D,MaxPooling2D
from keras.layers import BatchNormalization
from matplotlib import pyplot as plt
import numpy as np
import cPickle
import tensorflow as tf
import keras
import sys,os
# fix random seed for reproducibility
np.random.seed(7)

#hyperparameters used in CNN
batch_size = 32 
num_epochs = 50
kernel_size = 3 
pool_size = 2 
conv_depth_1 = 32 
conv_depth_2 = 64 
drop_prob_1 = 0.25 
drop_prob_2 = 0.5 
hidden_size = 512 

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def cnn(X_train,Y_train,activation,X_test):
        model = Sequential()
        model.add(Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same',
                         input_shape=X_train.shape[1:]))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Conv2D(conv_depth_1, (kernel_size, kernel_size),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size),dim_ordering='th'))
        model.add(Dropout(drop_prob_1))

        model.add(Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Conv2D(conv_depth_2, (kernel_size, kernel_size),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size),dim_ordering='th'))
        model.add(Dropout(drop_prob_1))

        model.add(Flatten())
        model.add(Dense(hidden_size))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(drop_prob_2))
        model.add(Dense(10))        #num_classes=10
        model.add(BatchNormalization())
        model.add(Activation('softmax'))
        optimizer = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
        model.fit(X_train, Y_train,                
            batch_size=batch_size, epochs=num_epochs,
            verbose=1, validation_split=0.2) 
        
        predicted_classes = model.predict(X_test)
        max_class = predicted_classes.argmax(axis=1)

        fil = open("q2_b_output.txt","w+")
        for i in range(len(max_class)):
            fil.write(label_names[max_class[i]]+"\n")

# def cnn(X_train,Y_train):  
#     inp = Input(shape=(3, 32, 32))
#     conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
#     conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
#     pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size),dim_ordering="th")(conv_2)
#     drop_1 = Dropout(drop_prob_1)(pool_1)
#     # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
#     conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
#     conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
#     pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size),dim_ordering="th")(conv_4)
#     drop_2 = Dropout(drop_prob_1)(pool_2)
#     # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
#     flat = Flatten()(drop_2)
#     hidden = Dense(hidden_size, activation='relu')(flat)
#     drop_3 = Dropout(drop_prob_2)(hidden)
#     out = Dense(10, activation='softmax')(drop_3)
#     model = Model(inputs=inp, outputs=out) 
#     model.compile(loss='categorical_crossentropy', 
#                 optimizer='adam', 
#                 metrics=['accuracy'])
#     model.fit(X_train, Y_train,                
#             batch_size=batch_size, epochs=num_epochs,
#             verbose=1, validation_split=0.2) 
    # model.evaluate(X_test, Y_test, verbose=1)  

if "__name__" != "__main__":
    train_dir = sys.argv[1]
    test_file = sys.argv[2]
    train_data = np.zeros([0,0])
    train_labels = np.zeros([0,0])
    label_names = []
    flag = 0
    for filename in os.listdir(train_dir):
        # print train_dir+filename
        a = unpickle(train_dir+filename)
        if not flag and not filename == "batches.meta":
            train_data = a["data"]
            train_labels = a["labels"]
            flag = 1
        elif flag and not filename == "batches.meta":
            train_data = np.concatenate((train_data,a["data"]),axis=0)
            train_labels = np.concatenate((train_labels,a["labels"]),axis=0)
        elif filename == "batches.meta":
            label_names = a["label_names"]
            
    #reshaping the training data and their classes
    # print label_names
    train_data = np.reshape(train_data,(train_data.shape[0], 3, 32, 32))
    train_labels = np_utils.to_categorical(train_labels, 10)
    test_data = unpickle(test_file)["data"]
    test_data = np.reshape(train_data,(train_data.shape[0], 3, 32, 32))
    train_data = train_data.astype('float32')
    train_data /= 255
    # print train_data.shape
    # print train_labels.shape
    # print label_names
    cnn(train_data,train_labels,'relu',test_data)