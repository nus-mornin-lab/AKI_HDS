from preprocess import *
import matplotlib.pyplot as plt
import datetime
import pandas as pd

def plottimeseries(features, patient_stay):
	time_series_normalized_filename = './all_time_series_normalized.csv'
	time_series_normalized_file = Path(time_series_normalized_filename)
	all_time_series_concat = pd.read_csv(time_series_normalized_filename)
	time_series = all_time_series_concat[all_time_series_concat['icustay_id']==patient_stay['icustay_id']]
	_, procedures = getAllEvents(patient_stay['hadm_id'])
	time_series['Time'] = pd.to_datetime(time_series['Time'])
	plt.figure()
	lines = []
	for feature in features:
		line, = plt.plot(time_series['Time'], time_series[feature], '-o', label=feature)
		lines.append(line)
	plt.legend(lines, features)
	if len(procedures):
		for index, procedure in procedures.iterrows():
			plt.scatter(procedure['starttime'], 0.5, s=200, c='r')
	if type(patient_stay['dod']) == str:
		dod = datetime.datetime.strptime(patient_stay['dod'][:10], '%Y-%m-%d')
		last = time_series['Time'].max()
		if dod <= last + datetime.timedelta(hours=1):
			plt.scatter(dod+datetime.timedelta(days=1), 0.5, s=200, c='k')
	plt.show()
