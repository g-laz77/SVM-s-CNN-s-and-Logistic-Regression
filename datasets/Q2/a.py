import numpy as np

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

if "__name__" != "__main__":

	'''
	Each of the batch files contains a dictionary with the following elements:

    1. data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. 
	The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 '
	the blue. The image is stored in row-major order, so that the first 32 entries of the array are the
	 red channel values of the first row of the image.
    2. labels -- a list of 10000 numbers in the range 0-9. 
	The number at index i indicates the label of the ith image in the array data.

	The  batches.meta file contains a Python dictionary object. It has the label names. 
	'''
	a = unpickle("data_batch_1")
	data = a["data"] 
	labels = a["labels"] 
	b = unpickle("batches.meta")
	label_names = b["label_names"]

	print data.shape
	print len(labels)
	print label_names