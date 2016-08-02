import numpy as np

class Batch(object):
	def __init__(self, feature, label):
		if(len(feature) != len(label)):
			print "[WARN]feature's size and label's size aren't same"
		
		self.feature_matrix = feature
		self.label_matrix = label
		self.length = len(feature)
		self.pointer = 0

	def next_batch(self, batch_size):
		if(self.pointer + batch_size > self.length-1):
			feature_batch = np.concatenate((self.feature_matrix[ self.pointer: ], self.feature_matrix[ :self.pointer+batch_size-self.length ]), axis=0)
			label_batch = np.concatenate((self.label_matrix[ self.pointer: ], self.label_matrix[ :self.pointer+batch_size-self.length ]), axis=0)

			self.pointer = self.pointer + batch_size - self.length
		else:
			feature_batch = self.feature_matrix[self.pointer:self.pointer+batch_size]
			label_batch = self.label_matrix[self.pointer:self.pointer+batch_size]
			
			self.pointer = self.pointer + batch_size
		return feature_batch, label_batch
