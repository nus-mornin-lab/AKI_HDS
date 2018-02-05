import numpy as np


class DataIterator:
    def __init__(self, timeSeriesList, ratio=8.9, numBuckets=5):
        self.size = len(timeSeriesList)
        self.numBuckets = numBuckets
        self.epochs = 0
        self.cursor = 0
        timeSeriesList.sort(key=lambda x: len(x))
        self.sequenceLengths = np.array([len(timeSeries) for timeSeries in timeSeriesList])
        # add padding
        maxLength = len(timeSeriesList[-1])
        self.data = np.zeros([len(timeSeriesList), maxLength, len(timeSeriesList[0].columns)], dtype=np.float64)
        for i, data_i in enumerate(self.data):
            data_i[:self.sequenceLengths[i]] = timeSeriesList[i].values
        self.musks = np.zeros([len(timeSeriesList), maxLength], dtype=np.int32)
        for i, musk in enumerate(self.musks):
            musk[:self.sequenceLengths[i]] = np.array(
                [1 if self.data[i, j, -1] == 0 else ratio for j in range(self.sequenceLengths[i])])

    def next_batch(self, n=None):
        if n is None:
            n = self.size
        index = min(self.cursor + n, self.size)
        maxLength = max(self.sequenceLengths[self.cursor:index])
        x = self.data[self.cursor:index, :maxLength, :-1]
        y = self.data[self.cursor:index, :maxLength, -1]
        sequenceLengths = self.sequenceLengths[self.cursor:index]
        musks = self.musks[self.cursor:index, :maxLength]
        self.cursor += n
        if self.cursor >= self.size:
            self.epochs += 1
            self.cursor = 0
        return x, y, sequenceLengths, musks
