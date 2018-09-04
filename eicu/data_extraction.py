from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
import pandas as pd
import numpy as np
from pathlib import Path
from connection import *
from datetime import datetime, timedelta
from features import *
import sys
import json
from collections import defaultdict
import time

features_json_data = open('./features.json').read()
features = json.loads(features_json_data)
# allFeatures = ['age', 'gender', 'Heart Rate',
#                'Respiratory Rate', 'SpO2/SaO2', 'pH', 'Potassium',
#                'Calcium', 'Glucose', 'Sodium', 'HCO3', 'White Blood Cells',
#                'Hemoglobin', 'Red Blood Cells', 'Platelet Count',
#                'Urea Nitrogen', 'Creatinine', 'Weight', 'gcs', 'ventilation',
#                'vasoactive medications', 'Blood Pressure', 'sedative medications', '1 hours urine output']

allFeatures = [
               'ventilation']

engine = getEngine()

def getICUStayPatients(engine=engine, force_reload=False):
    icustays_filepath = './selected_stays.csv'
    icustays_file = Path(icustays_filepath)
    if not force_reload and icustays_file.is_file():
        datecols = ['intime', 'outtime']
        icustays = pd.read_csv(icustays_filepath, parse_dates=datecols)
    else:
        icustays = pd.read_sql_query("""
            SELECT 
                  i.uniquepid, i.patientunitstayid, p.gender, 
                  substring(p.age FROM '[0-9]+') AS age, 
                  CASE WHEN lower(p.unitdischargestatus) like '%alive%' THEN 0
                       WHEN lower(p.unitdischargestatus) like '%expired%' THEN 1 
                  ELSE NULL END AS icu_mort,
                  p.unitadmityear, p.unitadmittime24, p.unitadmittime, 
                  p.unitdischargeyear, p.unitdischargetime24, p.unitdischargetime,
                  i.unitdischargeoffset, 
                  i.icu_los_hours, 
                  i.admissionweight AS weight
            FROM public.icustay_detail AS i
            LEFT JOIN patient AS p 
            ON i.patientunitstayid = p.patientunitstayid
            WHERE i.unitvisitnumber=1
            AND icu_los_hours>=0.5 AND icu_los_hours<=30
            AND p.unitdischargeoffset>=720;
            """, con=engine)
        # get patients having all the features we need
        patientunitstayids = getPatientsHavingAllFeatures()
        icustays = icustays[icustays['patientunitstayid'].isin(patientunitstayids)]
        # calculate the age
        #icustays['age'] = icustays['intime'] - icustays['dob']
        #icustays['age'] = (icustays['age'] / np.timedelta64(1, 'Y')).astype(int)
        icustays['age'] = icustays['age'].apply(pd.to_numeric)
        icustays = icustays[icustays['age'] >= 16]
        # convert gender to binary values
        icustays['gender'] = (icustays['gender'] == 'Male').astype(int)
        icustays['intime'] = pd.to_datetime(icustays.unitadmityear.astype(str) + ' ' + icustays.unitadmittime24.astype(str))
        icustays['outtime'] = pd.to_datetime(icustays['intime'] + pd.to_timedelta(icustays.unitdischargeoffset, unit='m'))
        icustays.reset_index(inplace=True, drop=True)
        icustays.to_csv(icustays_filepath, index=False)
    #for column in ('intime', 'outtime', 'dob', 'dod'):
    #    icustays[column] = pd.to_datetime(icustays[column])
    return icustays

def getPatientsHavingAllFeatures():
    # run up to 30 db queries at a time
    pool = ThreadPoolExecutor(30)
    filters = [pool.submit(getPatientsHavingFeature, feature) for feature in features if feature['label']]
    wait(filters)
    filters = [filter.result() for filter in filters]
    return set.intersection(*filters)

def getPatientsHavingFeature(feature):
    queryStr = """
    SELECT DISTINCT patientunitstayid FROM public.{table}
    WHERE {name} IS NOT NULL 
    AND {name} BETWEEN {min} AND {max}
    """
    patientunitstayid = set()
    engine = getEngine()
    for table in ('pivoted_lab', 'pivoted_vital', 'pivoted_bg', 'icustay_detail', 'pivoted_score'):
        if table not in feature: continue
        patientunitstayid |= set(pd.read_sql_query(
                queryStr.format(
                    table=table,
                    name=feature[table],
                    min=feature['min'],
                    max=feature['max']
                ),
                con=engine)['patientunitstayid'].tolist())
        #patientunitstayid = [i for i in patientunitstayid if not pd.read_sql_query(urineOutputQuery.format(patientunitstayid=i), con=engine).empty]
    return patientunitstayid

def roundTimeToNearestHour(dt):
    if dt.minute >= 30:
        dt += timedelta(minutes=30)
    return datetime(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour)

def getTimeStamps(patientStay):
    startTime = roundTimeToNearestHour(patientStay['intime'])
    # if patientStay['dod']:
    #     endTime = roundTimeToNearestHour(min(patientStay['outtime'], patientStay['dod']))
    # else:
    #     endTime = roundTimeToNearestHour(patientStay['outtime'])
    endTime = roundTimeToNearestHour(patientStay['outtime'])
    hour = timedelta(hours=1)
    gap = endTime - startTime
    gapInHours = gap.days*24+gap.seconds//3600
    return pd.Series([startTime+hour*i for i in range(gapInHours+1)])

def getchartoffset(patientStay, intime):
    return (patientStay - intime).astype('timedelta64[m]')

def fillNa(timeSeries):
    for feature in allFeatures:
        timeSeries[feature].interpolate(method='ffill', inplace=True)
        timeSeries[feature].fillna(method='ffill', inplace=True)
        timeSeries[feature].fillna(method='backfill', inplace=True)
    timeSeries.fillna(0, inplace=True)
    return timeSeries

def getPatientTimeSeries(patientStay):
    icustayID = patientStay['patientunitstayid']
    intime = patientStay['intime']
    #hadmID = patientStay['hadm_id']
    timeSeries = pd.DataFrame({'Time': getTimeStamps(patientStay)})
    timeSeries['intime'] = pd.Series([intime] * len(timeSeries))
    timeSeries['patientunitstayid'] = pd.Series([icustayID] * len(timeSeries))
    timeSeries['age'] = pd.Series([patientStay['age']] * len(timeSeries))
    timeSeries['gender'] = pd.Series([patientStay['gender']] * len(timeSeries))
    timeSeries['Weight'] = pd.Series([patientStay['weight']] * len(timeSeries))
    timeSeries['chartoffset'] = getchartoffset(timeSeries['Time'], intime)
    con = getEngine()
    # for feature in features:
    #      addFeature(feature, timeSeries, icustayID, con)
    # addGCS(icustayID, timeSeries, con)
    # addVasopressor(icustayID, timeSeries, con)
    # addSedative(icustayID, timeSeries, con)
    addVentilation(icustayID, timeSeries, con)
    # addUrineOutput(icustayID, timeSeries, con, windowSize=1, offset=0)
    # addUrineOutput(icustayID, timeSeries, con)
    # addNextNHoursUrineOutput(timeSeries, 6)
    # Pending
    # addProcedure({'view': 'rx', 'label': 'ventilation'}, icustayID, timeSeries, con)

    # Done
    # addProcedure({'view': 'vasopressordurations', 'label': 'vasoactive medications'}, icustayID, timeSeries, con)

    # addProcedure({'view': 'sedativedurations', 'label': 'sedative medications'}, icustayID, timeSeries, con)
    fillNa(timeSeries)
    timeSeries.drop(timeSeries.index[-6:], inplace=True)
    return timeSeries

def getAllPatientsTimeSeries(stays, forceReload=False):
    timeSeriesFilename = './time_series.csv'
    timeSeriesFile = Path(timeSeriesFilename)
    if not forceReload and timeSeriesFile.is_file():
        allTimeSeriesConcat = pd.read_csv(timeSeriesFile)
        allTimeSeries = [timeSeries.sort_values('Time') for icustayID, timeSeries in
                         allTimeSeriesConcat.groupby('patientunitstayid')]
        return allTimeSeries
    allTimeSeries = []
    for i in range(0, len(stays), 100):
        pool = ProcessPoolExecutor()
        batchTimeSeries = [pool.submit(getPatientTimeSeries, stays.iloc[j]) for j in range(i, min(i+100, len(stays)))]
        # for i in range(len(stays)):
        #     batchTimeSeries = getPatientTimeSeries(stays.iloc[i])
        wait(batchTimeSeries)
        batchTimeSeries = [timeSeries.result() for timeSeries in batchTimeSeries]
        allTimeSeries += batchTimeSeries
        if (i+100) % 1000 == 0:
            print("Extracted time series for first {n} patients".format(n=min(i+100, len(stays))))
    allTimeSeriesConcat = pd.concat(allTimeSeries)
    allTimeSeriesConcat.to_csv(timeSeriesFilename, index=False)
    return allTimeSeries

def normalizeFeatures(allTimeSeries, forceReload=False):
    fileName = './time_series_normalized.csv'
    file = Path(fileName)
    if not forceReload and file.is_file():
        allTimeSeriesConcat = pd.read_csv(file)
        allTimeSeries = [timeSeries.sort_values('Time') for icustayID, timeSeries in
                         allTimeSeriesConcat.groupby('patientunitstayid')]
        return allTimeSeries
    allTimeSeriesConcat = pd.concat(allTimeSeries)
    for feature in allFeatures:
        minValue = allTimeSeriesConcat[feature].min()
        maxValue = allTimeSeriesConcat[feature].max()
        diff = maxValue - minValue
        allTimeSeriesConcat[feature] = (allTimeSeriesConcat[feature]-minValue)/diff
    allTimeSeriesConcat.to_csv(fileName, index=False)
    allTimeSeries = [timeSeries.sort_values('Time') for icustayID, timeSeries in
                     allTimeSeriesConcat.groupby('patientunitstayid')]
    return allTimeSeries

if __name__ == "__main__":
    stays = getICUStayPatients()
    print("Select {n} icu stay patients".format(n=len(stays)))
    allTimeSeries = getAllPatientsTimeSeries(stays)
    print("Extracted time series for all patients.")
    allTimeSeries = normalizeFeatures(allTimeSeries)
    print("Normalized all features")