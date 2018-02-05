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

urineOutputQuery = """
    SELECT charttime,
    case when itemid = 227488 then -1*value else value end AS value
    FROM outputevents
    WHERE hadm_id={hadmID} AND 
    value IS NOT NULL AND
    itemid IN (
    -- these are the most frequently occurring urine output observations in CareVue
    40055, -- "Urine Out Foley"
    43175, -- "Urine ."
    40069, -- "Urine Out Void"
    40094, -- "Urine Out Condom Cath"
    40715, -- "Urine Out Suprapubic"
    40473, -- "Urine Out IleoConduit"
    40085, -- "Urine Out Incontinent"
    40057, -- "Urine Out Rt Nephrostomy"
    40056, -- "Urine Out Lt Nephrostomy"
    40405, -- "Urine Out Other"
    40428, -- "Urine Out Straight Cath"
    40086,--	Urine Out Incontinent
    40096, -- "Urine Out Ureteral Stent #1"
    40651, -- "Urine Out Ureteral Stent #2"
    
    -- these are the most frequently occurring urine output observations in MetaVision
    226559, -- "Foley"
    226560, -- "Void"
    226561, -- "Condom Cath"
    226584, -- "Ileoconduit"
    226563, -- "Suprapubic"
    226564, -- "R Nephrostomy"
    226565, -- "L Nephrostomy"
    226567, --	Straight Cath
    226557, -- R Ureteral Stent
    226558, -- L Ureteral Stent
    227488, -- GU Irrigant Volume In
    227489  -- GU Irrigant/Urine Volume Out
    );
    """
features_json_data = open('./features.json').read()
features = json.loads(features_json_data)
allFeatures = ['age', 'gender', 'Heart Rate',
                   'Respiratory Rate', 'SpO2/SaO2', 'pH', 'Potassium',
                   'Calcium', 'Glucose', 'Sodium', 'HCO3', 'White Blood Cells',
                   'Hemoglobin', 'Red Blood Cells', 'Platelet Count', 
                   'Urea Nitrogen', 'Creatinine', 'Weight', 'gcs', 'ventilation',
                   'vasoactive medications', 'Blood Pressure', 'sedative medications', '1 hours urine output']
engine = getEngine()


def getPatientsHavingFeature(feature):
    queryStr = """
    SELECT DISTINCT hadm_id FROM {table}
    WHERE itemid IN {itemids} AND valuenum IS NOT NULL
    AND valuenum  BETWEEN {min} AND {max}
    """
    hadmIDs = set()
    engine = getEngine()
    for table in ('chartevents', 'labevents'):
        if table not in feature: continue
        hadmIDs |= set(pd.read_sql_query(
                queryStr.format(
                    table=table,
                    itemids='(' + ','.join(list(map(str, feature[table]))) + ')',
                    min=feature['min'],
                    max=feature['max']
                ),
                con=engine)['hadm_id'].tolist())
    return hadmIDs


def getPatientsHavingAllFeatures():
    # run up to 30 db queries at a time
    pool = ThreadPoolExecutor(30)
    filters = [pool.submit(getPatientsHavingFeature, feature) for feature in features if feature['label']]
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
            AND i.outtime-i.intime>=interval '12' hour AND (p.dod IS NULL OR p.dod-i.intime>=interval '12' hour);
            """, con=engine)
        # get patients having all the features we need
        hadmIDs = getPatientsHavingAllFeatures()
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
            AND valuenum BETWEEN {min} AND {max}
            """
    if 'chartevents' in feature:
        charteventsQuery = query.format(table='chartevents',
                                        hadmID=hadmID,
                                        itemIDs='(' + ','.join(list(map(str, feature['chartevents']))) + ')',
                                        startTime=startTime,
                                        endTime=endTime,
                                        min=feature['min'],
                                        max=feature['max'])
    if 'labevents' in feature:
        labeventsQuery = query.format(table='labevents',
                                      hadmID=hadmID,
                                      itemIDs='(' + ','.join(list(map(str, feature['labevents']))) + ')',
                                      startTime=startTime,
                                      endTime=endTime,
                                      min=feature['min'],
                                      max=feature['max'])
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


def addUrineOutput(hadmID, timeSeries, con, windowSize=6, offset=1):
    urineOutputs = pd.read_sql(urineOutputQuery.format(hadmID=hadmID), con)
    urineOutputs['charttime'] = urineOutputs['charttime'].apply(roundTimeToNearestHour)
    validTimes = set(list(map(str, timeSeries['Time'].tolist()))[windowSize-1+offset:])
    timedeltas = [timedelta(hours=i) for i in range(offset, windowSize+offset)]
    urineOutputsSum = {time: 0 for time in validTimes}
    for _, row in urineOutputs.iterrows():
        value = row['value']
        for delta in timedeltas:
            time = row['charttime'] + delta
            if str(time) in validTimes:
                urineOutputsSum[str(time)] += value
    urineOutputColumn = pd.Series([urineOutputsSum[str(time)] if str(time) in urineOutputsSum else 0 for time in timeSeries['Time']])
    timeSeries['{n} hours urine output'.format(n=windowSize)] = urineOutputColumn


def addGCS(hadmID, timeSeries, con):
    query = """
    SELECT c1.charttime AS currenttime,
    case
        -- endotrach/vent is assigned a value of 0, later parsed specially
        when c1.itemid = 723 and c1.value = '1.0 ET/Trach' then 0 -- carevue
        when c1.itemid = 223900 and c1.value = 'No Response-ETT' then 0 -- metavision
        else c1.valuenum
    end AS value{label},
    c2.charttime AS prevtime,
    case
        -- endotrach/vent is assigned a value of 0, later parsed specially
        when c2.itemid = 723 and c2.value = '1.0 ET/Trach' then 0 -- carevue
        when c2.itemid = 223900 and c2.value = 'No Response-ETT' then 0 -- metavision
        else c2.valuenum
    end AS prevvalue{label} FROM chartevents AS c1 
    LEFT JOIN (
      SELECT c3.itemid, c3.value, c3.charttime, c3.valuenum
      FROM chartevents AS c3
      WHERE c3.itemid IN {itemIDs} AND c3.hadm_id={hadmID}
    ) AS c2
    ON c2.charttime<c1.charttime AND c2.charttime>c1.charttime-interval '6' hour
    WHERE c1.itemid IN {itemIDs} AND hadm_id={hadmID}
    ORDER BY currenttime DESC, prevtime DESC;
    """
    motor = pd.read_sql(query.format(hadmID=hadmID, itemIDs='(454,223901)', label='motor'), con)
    verbal = pd.read_sql(query.format(hadmID=hadmID, itemIDs='(723,223900)', label='verbal'), con)
    eyes = pd.read_sql(query.format(hadmID=hadmID, itemIDs='(184,220739)', label='eyes'), con)
    for df in (motor, verbal, eyes):
        existingTimes = set()
        indexToDrop = []
        for i, row in df.iterrows():
            if str(row['currenttime']) in existingTimes:
                indexToDrop.append(i)
            else:
                existingTimes.add(str(row['currenttime']))
        df.drop(df.index[indexToDrop], inplace=True)
        df.drop('prevtime', axis=1,inplace=True)
    df = pd.merge(motor, pd.merge(verbal, eyes, how='outer', on=['currenttime']), how='outer', on=['currenttime'])
    def coalesce(l):
        return next(item for item in l if item is not None)
    def calculateGCS(row):
        if row['valueverbal'] == 0:
            return 15
        if row['valueverbal'] is None and row['prevvalueverbal'] == 0:
            return 15
        if row['prevvalueverbal'] == 0:
            return coalesce([row['valuemotor'], 6]) +\
                   coalesce([row['valueverbal'], 5]) +\
                   coalesce([row['valueeyes'], 4])

        return coalesce([row['valuemotor'], row['prevvaluemotor'], 6]) + \
               coalesce([row['valueverbal'], row['prevvalueverbal'], 5]) + \
               coalesce([row['valueeyes'], row['prevvalueeyes'], 4])
    if len(df) == 0:
        timeSeries['gcs'] = pd.Series([np.nan for _ in range(len(timeSeries))])
        return
    df['gcs'] = df.apply(calculateGCS, axis=1)
    df['currenttime'] = df['currenttime'].apply(roundTimeToNearestHour)
    timeToGCS = {str(row['currenttime']): row['gcs'] for _, row in df.iterrows()}
    timeSeries['gcs'] = pd.Series([timeToGCS[str(time)] if str(time) in timeToGCS else np.nan for time in timeSeries['Time']])


def addProcedure(procedure, icustayID, timeSeries, con):
    procedures = pd.read_sql("""
    SELECT starttime, endtime FROM {view} WHERE icustay_id={icustayID};
    """.format(view=procedure['view'], icustayID=icustayID), con)
    timeSeries[procedure['label']] = pd.Series([1 if any((row['starttime'] <= time <= row['endtime']
                         for _, row in procedures.iterrows()))
               else 0 for time in timeSeries['Time']])


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


def addNextNHoursUrineOutput(timeSeries, n=6):
    timeSeries['Next {n} hours urine output'.format(n=n)] = \
        list(timeSeries['{n} hours urine output'.format(n=n)].iloc[n:]) + [0]*n
    timeSeries['AKI'] = (
        timeSeries['Next {n} hours urine output'.format(n=n)] / timeSeries['Weight'] / 6 <= 0.5).astype(np.int)


def fillNa(timeSeries):
    for feature in allFeatures:
        timeSeries[feature].interpolate(method='ffill', inplace=True)
        timeSeries[feature].fillna(method='ffill', inplace=True)
        timeSeries[feature].fillna(method='backfill', inplace=True)
    timeSeries.fillna(0, inplace=True)
    return timeSeries


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
    addUrineOutput(hadmID, timeSeries, con, windowSize=1, offset=0)
    addUrineOutput(hadmID, timeSeries, con)
    addNextNHoursUrineOutput(timeSeries, 6)
    addGCS(hadmID, timeSeries, con)
    addProcedure({'view': 'ventdurations', 'label': 'ventilation'}, icustayID, timeSeries, con)
    addProcedure({'view': 'vasopressordurations', 'label': 'vasoactive medications'}, icustayID, timeSeries, con)
    addProcedure({'view': 'sedativedurations', 'label': 'sedative medications'}, icustayID, timeSeries, con)
    fillNa(timeSeries)
    timeSeries.drop(timeSeries.index[-6:], inplace=True)
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


def addNotNullColumns(timeSeries):
    for feature in allFeatures:
        timeSeries[feature + ' not null'] = pd.notnull(timeSeries[feature]).astype(np.int32)
    return timeSeries


def addNotNullColumnsForAllPatients(allTimeSeries, forceReload=False):
    fileName = './time_series_with_not_null.csv'
    file = Path(fileName)
    if not forceReload and file.is_file():
        allTimeSeriesConcat = pd.read_csv(file)
        allTimeSeries = [timeSeries.sort_values('Time') for icustayID, timeSeries in
                         allTimeSeriesConcat.groupby('icustay_id')]
        return allTimeSeries
    allTimeSeriesWithNotNullColumns = []
    for i in range(0, len(allTimeSeries), 100):
        pool = ProcessPoolExecutor()
        batchTimeSeries = [pool.submit(addNotNullColumns, allTimeSeries[j]) for j in range(i, min(i+100, len(allTimeSeries)))]
        wait(batchTimeSeries)
        batchTimeSeries = [timeSeries.result() for timeSeries in batchTimeSeries]
        allTimeSeriesWithNotNullColumns += batchTimeSeries
        if (i+100) % 1000 == 0:
            print("Added not null columns for {n} patients.".format(n=min(i+100, len(allTimeSeries))))
    allTimeSeriesConcat = pd.concat(allTimeSeriesWithNotNullColumns)
    allTimeSeriesConcat.to_csv(fileName, index=False)
    return allTimeSeriesWithNotNullColumns


def normalizeFeatures(allTimeSeries, forceReload=False):
    fileName = './time_series_normalized.csv'
    file = Path(fileName)
    if not forceReload and file.is_file():
        allTimeSeriesConcat = pd.read_csv(file)
        allTimeSeries = [timeSeries.sort_values('Time') for icustayID, timeSeries in
                         allTimeSeriesConcat.groupby('icustay_id')]
        return allTimeSeries
    allTimeSeriesConcat = pd.concat(allTimeSeries)
    for feature in allFeatures:
        minValue = allTimeSeriesConcat[feature].min()
        maxValue = allTimeSeriesConcat[feature].max()
        diff = maxValue - minValue
        allTimeSeriesConcat[feature] = (allTimeSeriesConcat[feature]-minValue)/diff
    allTimeSeriesConcat.to_csv(fileName, index=False)
    allTimeSeries = [timeSeries.sort_values('Time') for icustayID, timeSeries in
                     allTimeSeriesConcat.groupby('icustay_id')]
    return allTimeSeries


if __name__ == "__main__":
    stays = getICUStayPatients()
    print("Select {n} icu stay patients".format(n=len(stays)))
    allTimeSeries = getAllPatientsTimeSeries(stays)
    print("Extracted time series for all patients.")
    allTimeSeries = normalizeFeatures(allTimeSeries)
    print("Normalized all features")
