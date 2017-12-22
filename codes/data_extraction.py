from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
import pandas as pd
import numpy as np
from pathlib import Path
from connection import *
from datetime import datetime, timedelta
import sys
import json
from collections import defaultdict
import time

features_json_data = open('./features.json').read()
features = json.loads(features_json_data)
mustHaveFeatures = set(['Heart Rate', 'Red Blood Cells', 'Urea Nitrogen', 'Hemoglobin', 'Platelet Count',
                        'Respiratory Rate', 'PaO2', 'paCO2', 'Creatinine', 'HCO3', 'Chloride', 'Calcium',
                        'White Blood Cells', 'pH', 'Potassium', 'Sodium', 'SpO2/SaO2', 'Glucose'])
engine = getEngine()


def getPatientsHavingFeature(feature):
    queryStr = 'SELECT DISTINCT hadm_id FROM {table} WHERE itemid IN {itemids} AND valuenum IS NOT NULL'
    hadmIDs = set()
    engine = getEngine()
    for table in ('chartevents', 'labevents'):
        if table not in feature: continue
        hadmIDs |= set(pd.read_sql_query(
                queryStr.format(
                    table=table,
                    itemids='(' + ','.join(list(map(str, feature[table]))) + ')'
                ),
                con=engine)['hadm_id'].tolist())
    return hadmIDs


def getPatientsHavingAllMustHaveFeatures():
    # run up to 30 db queries at a time
    pool = ThreadPoolExecutor(30)
    filters = [pool.submit(getPatientsHavingFeature, feature) for feature in features if feature['label'] in mustHaveFeatures]
    wait(filters)
    filters = [filter.result() for filter in filters]
    return set.intersection(*filters)


def getICUStayPatients(engine=engine, force_reload=False):
    icustays_filepath = './selected_stays.csv'
    icustays_file = Path(icustays_filepath)
    if not force_reload and icustays_file.is_file():
        icustays = pd.read_csv(icustays_filepath)
    else:
        icustays = pd.read_sql_query("""
            SELECT i.hadm_id, i.icustay_id, p.gender, p.dob, i.intime, i.outtime, p.dod, i.los
            FROM icustays AS i
            LEFT JOIN patients AS p ON i.subject_id=p.subject_id
            WHERE first_wardid=last_wardid
            AND first_careunit=last_careunit
            AND los>=0.5 AND los<=30
            AND i.intime<i.outtime AND i.intime<p.dod;
            """, con=engine)
        # get patients having all the features we need
        hadmIDs = getPatientsHavingAllMustHaveFeatures()
        icustays = icustays[icustays['hadm_id'].isin(hadmIDs)]
        # calculate the age
        icustays['age'] = icustays['intime'] - icustays['dob']
        icustays['age'] = (icustays['age'] / np.timedelta64(1, 'Y')).astype(int)
        icustays = icustays[icustays['age'] >= 16]
        # convert gender to binary values
        icustays['gender'] = (icustays['gender'] == 'M').astype(int)
        icustays.reset_index(inplace=True, drop=True)
        icustays.to_csv(icustays_filepath, index=False)
    for column in ('intime', 'outtime', 'dob', 'dod'):
        icustays[column] = pd.to_datetime(icustays[column])
    return icustays


def roundTimeToNearestHour(dt):
    if dt.minute >= 30:
        dt += timedelta(minutes=30)
    return datetime(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour)


def addFeature(feature, hadmID, timeSeries, con):
    startTime = timeSeries['Time'][0]
    endTime = timeSeries['Time'][len(timeSeries)-1]
    charteventsQuery = labeventsQuery = None
    query = """
            SELECT valuenum, charttime FROM {table}
            WHERE hadm_id={hadmID} AND itemid IN {itemIDs}
            AND charttime>='{startTime}' AND charttime<='{endTime}'
            AND valuenum IS NOT NULL
            """
    if 'chartevents' in feature:
        charteventsQuery = query.format(table='chartevents', hadmID=hadmID, itemIDs='(' + ','.join(list(map(str, feature['chartevents']))) + ')', startTime=startTime, endTime=endTime)
    if 'labevents' in feature:
        labeventsQuery = query.format(table='labevents', hadmID=hadmID, itemIDs='(' + ','.join(list(map(str, feature['labevents']))) + ')', startTime=startTime, endTime=endTime)
    if charteventsQuery and labeventsQuery:
        query = charteventsQuery + 'UNION' + labeventsQuery
    else:
        query = charteventsQuery if charteventsQuery else labeventsQuery
    query += ';'
    chartAndLabEvents = pd.read_sql(query, con=con)
    chartAndLabEvents['charttime'] = chartAndLabEvents['charttime'].apply(roundTimeToNearestHour)
    recordsSum = defaultdict(float)
    recordsCount = defaultdict(int)
    for _, row in chartAndLabEvents.iterrows():
        timeStr = str(row['charttime'])
        recordsSum[timeStr] += row['valuenum']
        recordsCount[timeStr] += 1
    featureColumn = pd.Series([recordsSum[str(time)]/recordsCount[str(time)] if str(time) in recordsSum else np.nan for time in timeSeries['Time']])
    timeSeries[feature['label']] = featureColumn


def getTimeStamps(patientStay):
    startTime = roundTimeToNearestHour(patientStay['intime'])
    if patientStay['dod']:
        endTime = roundTimeToNearestHour(min(patientStay['outtime'], patientStay['dod']))
    else:
        endTime = roundTimeToNearestHour(patientStay['outtime'])
    hour = timedelta(hours=1)
    gap = endTime - startTime
    gapInHours = gap.days*24+gap.seconds//3600
    return pd.Series([startTime+hour*i for i in range(gapInHours+1)])


def getPatientTimeSeries(patientStay):
    icustayID = patientStay['icustay_id']
    hadmID = patientStay['hadm_id']
    timeSeries = pd.DataFrame({'Time': getTimeStamps(patientStay)})
    timeSeries['icustay_id'] = pd.Series([icustayID] * len(timeSeries))
    timeSeries['hadm_id'] = pd.Series([hadmID] * len(timeSeries))
    timeSeries['age'] = pd.Series([patientStay['age']] * len(timeSeries))
    timeSeries['gender'] = pd.Series([patientStay['gender']] * len(timeSeries))
    con = getEngine()
    for feature in features:
        addFeature(feature, hadmID, timeSeries, con)
    return timeSeries


def getAllPatientsTimeSeries(stays, forceReload=False):
    timeSeriesFilename = './time_series.csv'
    timeSeriesFile = Path(timeSeriesFilename)
    if not forceReload and timeSeriesFile.is_file():
        allTimeSeriesConcat = pd.read_csv(timeSeriesFile)
        allTimeSeries = [timeSeries.sort_values('Time') for icustayID, timeSeries in
                         allTimeSeriesConcat.groupby('icustay_id')]
        return allTimeSeries
    allTimeSeries = []
    for i in range(0, len(stays), 100):
        pool = ProcessPoolExecutor()
        batchTimeSeries = [pool.submit(getPatientTimeSeries, stays.iloc[j]) for j in range(i, min(i+100, len(stays)))]
        wait(batchTimeSeries)
        batchTimeSeries = [timeSeries.result() for timeSeries in batchTimeSeries]
        allTimeSeries += batchTimeSeries
        if (i+100) % 1000 == 0:
            print("Extracted time series for first {n} patients".format(n=min(i+100, len(stays))))
    allTimeSeriesConcat = pd.concat(allTimeSeries)
    allTimeSeriesConcat.to_csv(timeSeriesFilename, index=False)
    return allTimeSeries


if __name__ == "__main__":
    stays = getICUStayPatients()
    print("Select {n} icu stay patients".format(n=len(stays)))
    allTimeSeries = getAllPatientsTimeSeries(stays)
    print("Extracted time series for all patients.")
