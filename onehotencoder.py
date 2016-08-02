import numpy as np

def encode(label, maxValue):
	length = len(label)
	matrix = np.zeros([length, maxValue])

	for index in range(length):
		matrix[index][int(label[index])] = 1

	return matrix
