"""
Course: Statistical Methods in Artificial Intelligence (CSE471)
Semester: Fall '17
Professor: Gandhi, Vineet

Assignment 2: SVM using scikit-learn.
Skeleton code for implementing SVM classifier for a
character recognition dataset having precomputed features for
each character. This is your submission file.

Dataset is taken from: https://archive.ics.uci.edu/ml/datasets/letter+recognition

Remember
--------
1) SVM algorithms are not scale invariant.
"""
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold.t_sne import TSNE

import argparse, os, sys

def get_input_data(filename):
    """
    Function to read the input data from the letter recognition data file.

    Parameters
    ----------
    filename: The path to input data file

    Returns
    -------
    X: The input for the SVM classifier of the shape [n_samples, n_features].
       n_samples is the number of data points (or samples) that are to be loaded.
       n_features is the length of feature vector for each data point (or sample).
    Y: The labels for each of the input data point (or sample). Shape is [n_samples,].

    """

    X = []; Y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            Y.append(line[0])
            X.append([float(x) for x in line[1:]])
    X = np.asarray(X); Y = np.asarray(Y)

    """
    An important part is missing here. Corresponding to point (1) in "Remember".
    ===========================================================================
    """
    # YOUR CODE GOES HERE
    var = []
    X_mean = np.sum(X, axis=0)
    X_mean = np.divide(X_mean, X.shape[0])

    for i in range(X.shape[1]):
        sum = np.divide(np.sum([(np.subtract(X[j][i], X_mean[i])) ** 2 for j in range(X.shape[0])]), X.shape[0])
        var.append(sum)
    var = np.asarray(var)

    for i in range(X.shape[0]):
        X[i] = np.divide(np.subtract(X[i], X_mean), var)
    """
    ===========================================================================
    """

    return X, Y

def calculate_metrics(predictions, labels):
    """
    Function to calculate the precision, recall and F-1 score.

    Parameters
    ----------
    predictions: The predictions obtained as output from the SVM classifier
    labels: The true label values corresponding to the entries in predictions

    Returns
    -------
    precision: true_positives / (true_positives + false_positives)
    recall: true_positives / (true_positives + false_negatives)
    f1: 2 * (precision * recall) / (precision + recall)
    ===========================================================================
    """

    # YOUR CODE GOES HERE
    precision = precision_score(labels,predictions,average='macro')
    recall = recall_score(labels,predictions,average='macro')
    f1 = f1_score(labels,predictions,average='macro')
    """
    ===========================================================================
    """

    return precision, recall, f1

def calculate_accuracy(predictions, labels):
    """
    Function to calculate the accuracy for a given set of predictions and
    corresponding labels.

    Parameters
    ----------
    predictions: The predictions obtained as output from the SVM classifier
    labels: The true label values corresponding to the entries in predictions

    Returns
    -------
    accuracy: Fraction of total samples that have correct predictions (same as
    true label)

    """
    return accuracy_score(labels, predictions)

def SVM(train_data,
        train_labels,
        test_data,
        test_labels,
        kernel='rbf'):
    """
    Function to create, train and test the one-vs-all SVM using scikit-learn.

    Parameters
    ----------
    train_data: Numpy ndarray of shape [n_train_samples, n_features]
    train_labels: Numpy ndarray of shape [n_train_samples,]
    test_data: Numpy ndarray of shape [n_test_samples, n_features]
    test_labels: Numpy ndarray of shape [n_test_samples,]
    kernel: linear (default)
            Which kernel to use for the SVM

    Returns
    -------
    accuracy: Accuracy of the model on the test data
    top_predictions: Top predictions for each test sample
    precision: The precision score for the test data
    recall: The recall score for the test data
    f1: The F1-score for the test data

    """

    """
    Create an SVM instance with the required parameters and train it.
    For details on how to do this in scikit-learn, refer:
        http://scikit-learn.org/stable/modules/svm.html
    ==========================================================================
    """

    # YOUR CODE GOES HERE
    model = svm.SVC(C=100,kernel=kernel,gamma=0.1)
    model.fit(train_data,train_labels)
    train_predictions = model.predict(train_data)
    # print train_predictions
    """
    ==========================================================================
    """

    """
    Calculates training accuracy. Replace predictions and labels with your
    respective variable names.
    """
    train_accuracy = calculate_accuracy(train_predictions, train_labels)
    print "Training Accuracy: %.4f" % (train_accuracy)

    """
    Use the trained model to perform testing. Using the output of the testing
    prodecure, get the top prediction for each sample and calculate the accuracy
    on test data using the function given (as shown above for train accuracy).

    Also, complete the function given above for metrics using scikit-learn and
    return their values in this function.
    ==========================================================================
    """

    # YOUR CODE GOES HERE
    test_predictions = model.predict(test_data)
    # print test_predictions
    test_accuracy = calculate_accuracy(test_predictions, test_labels)
    print "Testing Accuracy: %.4f" % (test_accuracy)

    precision, recall, f1 = calculate_metrics(test_predictions, test_labels)
    """
    ==========================================================================
    """

    return test_accuracy, precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
            help='path to the directory containing the dataset file')

    args = parser.parse_args()
    if args.data_dir is None:
        print "Usage: python letter_classification_svm.py --data_dir='<dataset dir path>'"
        sys.exit()
    else:
        filename = os.path.join(args.data_dir, 'letter_classification_train.data')
        try:
            if os.path.exists(filename):
                print "Using %s as the dataset file" % filename
        except:
            print "%s not present in %s. Please enter the correct dataset directory" % (filename, args.data_dir)
            sys.exit()

    # Set the value for svm_kernel as required.
    svm_kernel = 'rbf'

    """
    Get the input data using the provided function. Store the X and Y returned
    as X_data and Y_data. Use filename found above as the input to the function.
    ==========================================================================
    """
    # YOUR CODE GOES HERE
    X_data, Y_data = get_input_data(filename)

    """
    ==========================================================================
    """

    """
    We use 5-fold cross validation for reporting the final scores and metrics.
    Code to generate the 5-folds (You can change the split function to something
    else that you like as well) and then calling the SVM to classify the present
    split.
    ==========================================================================
    """
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.20)   # Do not change this split size
    accumulated_metrics = []
    fold = 1
    for train_indices, test_indices in sss.split(X_data, Y_data):
        print "Fold%d -> Number of training samples: %d | Number of testing "\
            "samples: %d" % (fold, len(train_indices), len(test_indices))
        train_data, test_data = X_data[train_indices], X_data[test_indices]
        train_labels, test_labels = Y_data[train_indices], Y_data[test_indices]
        accumulated_metrics.append(
            SVM(train_data, train_labels, test_data, test_labels,
                svm_kernel))
        fold += 1

    """
    Print out the accumulated metrics in a good format.
    ==========================================================================
    """

    # YOUR CODE GOES HERE
    temp1 = temp2 = temp3 = temp4 = 0.0
    for i in range(len(accumulated_metrics)):
        temp1 += accumulated_metrics[i][0]/5
        temp2 += accumulated_metrics[i][1]/5
        temp3 += accumulated_metrics[i][2]/5
        temp4 += accumulated_metrics[i][3]/5
        
    print("Accuracy:",temp1)
    print("Precision:",temp2)
    print("Recall:",temp3)
    print("F1:",temp4)
    

    """
    ==========================================================================
    """
