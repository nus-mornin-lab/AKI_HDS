import numpy as np


class DataIterator:
	def __init__(self, timeSeriesList, numBuckets=5):
		self.size = len(timeSeriesList)
		self.numBuckets = numBuckets
		self.epochs = 0
		self.cursor = np.array([0] * numBuckets)
		timeSeriesList.sort(key=lambda x: len(x))
		self.sequenceLengths = np.array([len(timeSeries) for timeSeries in timeSeriesList])
		# add padding
		maxLength = len(max(timeSeriesList, key=lambda x: len(x)))
		self.data = np.zeros([len(timeSeriesList), maxLength, len(timeSeriesList[0].columns)], dtype=np.float64)
		for i, data_i in enumerate(self.data):
			data_i[:self.sequenceLengths[i]] = timeSeriesList[i].values
		self.musks = np.zeros([len(timeSeriesList), maxLength], dtype=np.int32)
		for i, musk in enumerate(self.musks):
			musk[:self.sequenceLengths[i]] = np.ones(self.sequenceLengths[i])

	def next_batch(self, n):
		if np.any(self.cursor + n + 1 > self.size):
			self.epochs += 1
			self.cursor -= self.cursor
		i = np.random.randint(0, self.numBuckets)
		index = min(self.cursor[i]+n, self.size)
		maxLength = max(self.sequenceLengths[self.cursor[i]:index])
		x = self.data[self.cursor[i]:index, :maxLength, :-1]
		y = self.data[self.cursor[i]:index, :maxLength, -1]
		sequenceLengths = self.sequenceLengths[self.cursor[i]:index]
		musks = self.musks[self.cursor[i]:index, :maxLength]
		self.cursor[i] += n
		return x, y, sequenceLengths, musks
