import numpy as np


class DataIterator:
	def __init__(self, timeSeriesList):
		self.cursor = 0
		self.size = len(timeSeriesList)
		timeSeriesList.sort(key=lambda x: len(x))
		self.sequenceLengths = [len(timeSeries) for timeSeries in timeSeriesList]
		# add padding
		maxLength = max(timeSeriesList, key=lambda x: len(x))
		self.data = np.zeros([len(timeSeriesList), maxLength, len(timeSeriesList[0].columns)], dtype=np.int32)
		for i, data_i in enumerate(self.data):
			data_i[:self.sequenceLengths[i]] = timeSeriesList[i].values

	def next_batch(self, n):
		index = min(self.cursor+n, self.size)
		maxLength = max(self.sequenceLengths[self.cursor:index])
		x = self.data[self.cursor:index, :maxLength, :-1]
		y = self.data[self.cursor:index, :maxLength, -1]
		self.cursor += n
		if self.cursor >= self.size:
			self.cursor = 0
		return x, y
