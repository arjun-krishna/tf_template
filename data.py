"""
@author: Arjun Krishna
@desc: data manager
"""
import numpy as np

class DataManager:

	def __init__(self, batch_size=64):
		self.batch_size = batch_size

	def get_batch(self):
		X = np.random.rand(self.batch_size, 20)
		y = np.zeros((self.batch_size, 10))
		for i in range(self.batch_size):
			y[i][0] = 1
		return X, y
