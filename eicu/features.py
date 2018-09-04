from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


urineOutputQuery = """
SELECT chartoffset, urineoutput FROM public.pivoted_uo
WHERE patientunitstayid={patientunitstayid}
AND urineoutput IS NOT NULL
AND chartoffset >= 0
"""

def roundTimeToNearestHour(dt):
    if dt.minute >= 30:
        dt += timedelta(minutes=30)
    return datetime(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour)

def addFeature(feature, timeSeries, patientunitstayid, con):
    startoffset = timeSeries['chartoffset'][0]
    endoffset = timeSeries['chartoffset'][len(timeSeries)-1]
    vitalQuery = labQuery = bgQuery = None
    query = """
            SELECT {feature}, chartoffset FROM {table}
            WHERE patientunitstayid={patientunitstayid} 
            AND chartoffset >= '{startoffset}'::decimal AND chartoffset <= '{endoffset}'::decimal 
            AND {feature} IS NOT NULL 
            AND {feature} BETWEEN {min} and {max}
            """

    if 'pivoted_vital' in feature:
        vitalQuery = query.format(table='public.pivoted_vital',
                                  patientunitstayid=patientunitstayid,
                                  feature=feature['pivoted_vital'],
                                  startoffset=startoffset,
                                  endoffset=endoffset,
                                  min=feature['min'],
                                  max=feature['max'])

    if 'pivoted_lab' in feature:
        vitalQuery = query.format(table='public.pivoted_lab',
                                patientunitstayid=patientunitstayid,
                                feature=feature['pivoted_lab'],
                                startoffset=startoffset,
                                endoffset=endoffset,
                                min=feature['min'],
                                max=feature['max'])

    if 'pivoted_bg' in feature:
        vitalQuery = query.format(table='public.pivoted_bg',
                                  patientunitstayid=patientunitstayid,
                                  feature=feature['pivoted_bg'],
                                  startoffset=startoffset,
                                  endoffset=endoffset,
                                  min=feature['min'],
                                  max=feature['max'])
    if 'pivoted_score' in feature:
        vitalQuery = query.format(table='public.pivoted_score',
                                  patientunitstayid=patientunitstayid,
                                  feature=feature['pivoted_score'],
                                  startoffset=startoffset,
                                  endoffset=endoffset,
                                  min=feature['min'],
                                  max=feature['max'])

    # if vitalQuery and labQuery:
    #     query = vitalQuery + 'UNION' + labQuery
    # else:
    #     query = labQuery if labQuery else vitalQuery
    query = vitalQuery
    query += ';'
    chartAndLabEvents = pd.read_sql_query(query, con=con)
    intime = pd.Series([timeSeries['intime'][0]] * len(chartAndLabEvents))
    chartAndLabEvents['charttime'] = pd.to_datetime(pd.to_datetime(intime) + pd.to_timedelta(chartAndLabEvents['chartoffset'], unit='m'))
    chartAndLabEvents['charttime'] = chartAndLabEvents['charttime'].apply(roundTimeToNearestHour)
    recordsSum = defaultdict(float)
    recordsCount = defaultdict(int)
    for _, row in chartAndLabEvents.iterrows():
        timeStr = str(row['charttime'])
        if 'pivoted_vital' in feature:
            recordsSum[timeStr] += row[feature['pivoted_vital']]
        elif 'pivoted_lab' in feature:
            recordsSum[timeStr] += row[feature['pivoted_lab']]
        elif 'pivoted_score' in feature:
            recordsSum[timeStr] += row[feature['pivoted_score']]
        else:
            recordsSum[timeStr] += row[feature['pivoted_bg']]
        recordsCount[timeStr] += 1
    featureColumn = pd.Series([recordsSum[str(time)]/recordsCount[str(time)] if str(time) in recordsSum else np.nan for time in timeSeries['Time']])
    timeSeries[feature['label']] = featureColumn

def addUrineOutput(patientunitstayid, timeSeries, con, windowSize=6, offset=1):
    urineOutputs = pd.read_sql_query(urineOutputQuery.format(patientunitstayid=patientunitstayid), con=con)
    intime = pd.Series([timeSeries['intime'][0]] * len(urineOutputs))
    urineOutputs['charttime'] = pd.to_datetime(pd.to_datetime(intime) + pd.to_timedelta(urineOutputs['chartoffset'], unit='m'))
    urineOutputs['charttime'] = urineOutputs['charttime'].apply(roundTimeToNearestHour)
    validTimes = set(list(map(str, timeSeries['Time'].tolist()))[windowSize-1+offset:])
    timedeltas = [timedelta(hours=i) for i in range(offset, windowSize+offset)]
    urineOutputsSum = {time: 0 for time in validTimes}
    #print(validTimes)
    for _, row in urineOutputs.iterrows():
        value = row['urineoutput']
        for delta in timedeltas:
            #print(delta)
            time = row['charttime'] + delta
            #print(time)
            #print()
            if str(time) in validTimes:
                urineOutputsSum[str(time)] += value
    urineOutputColumn = pd.Series([urineOutputsSum[str(time)] if str(time) in urineOutputsSum else 0 for time in timeSeries['Time']])
    timeSeries['{n} hours urine output'.format(n=windowSize)] = urineOutputColumn

def addNextNHoursUrineOutput(timeSeries, n=6):
    timeSeries['Next {n} hours urine output'.format(n=n)] = \
        list(timeSeries['{n} hours urine output'.format(n=n)].iloc[n:]) + [0]*n
    timeSeries['AKI'] = (
        timeSeries['Next {n} hours urine output'.format(n=n)] / timeSeries['Weight'] / 6 <= 0.5).astype(np.int)

def addGCS(patientunitstayid, timeSeries, con):
    feature = {
    "label": "gcs",
    "pivoted_score": "gcs",
    "max": 999999.0,
    "min": 0.0
  }
    startoffset = timeSeries['chartoffset'][0]
    endoffset = timeSeries['chartoffset'][len(timeSeries) - 1]
    vitalQuery = None
    query = """
                SELECT {feature}, chartoffset FROM {table}
                WHERE patientunitstayid={patientunitstayid} 
                AND chartoffset >= '{startoffset}'::decimal AND chartoffset <= '{endoffset}'::decimal 
                AND {feature} IS NOT NULL 
                AND {feature} BETWEEN {min} and {max}
                """

    vitalQuery = query.format(table='public.pivoted_score',
                              patientunitstayid=patientunitstayid,
                              feature=feature['pivoted_score'],
                              startoffset=startoffset,
                              endoffset=endoffset,
                              min=feature['min'],
                              max=feature['max'])

    query = vitalQuery
    query += ';'
    chartAndLabEvents = pd.read_sql_query(query, con=con)
    intime = pd.Series([timeSeries['intime'][0]] * len(chartAndLabEvents))
    chartAndLabEvents['charttime'] = pd.to_datetime(
        pd.to_datetime(intime) + pd.to_timedelta(chartAndLabEvents['chartoffset'], unit='m'))
    chartAndLabEvents['charttime'] = chartAndLabEvents['charttime'].apply(roundTimeToNearestHour)
    recordsSum = defaultdict(float)
    recordsCount = defaultdict(int)
    for _, row in chartAndLabEvents.iterrows():
        timeStr = str(row['charttime'])
        recordsSum[timeStr] += row[feature['pivoted_score']]
        recordsCount[timeStr] += 1
    featureColumn = pd.Series(
        [recordsSum[str(time)] / recordsCount[str(time)] if str(time) in recordsSum else np.nan for time in
         timeSeries['Time']])
    timeSeries[feature['label']] = featureColumn

def addVasopressor(patientunitstayid, timeSeries, con):
    startoffset = timeSeries['chartoffset'][0]
    endoffset = timeSeries['chartoffset'][len(timeSeries) - 1]

    query = """
    SELECT chartoffset, vasopressor FROM public.pivoted_treatment_vasopressor
    WHERE patientunitstayid={patientunitstayid} 
    AND chartoffset >= '{startoffset}'::decimal AND chartoffset <= '{endoffset}'::decimal;
    """

    vassopressorChart = pd.read_sql_query(query.format(patientunitstayid=patientunitstayid, startoffset=startoffset, endoffset=endoffset), con=con)
    timeSeries['vasoactive medications'] = pd.Series(0 if vassopressorChart.empty else 1 for time in timeSeries['Time'])

def addSedative(patientunitstayid, timeSeries, con):
    startoffset = timeSeries['chartoffset'][0]
    endoffset = timeSeries['chartoffset'][len(timeSeries) - 1]

    query = """
        SELECT chartoffset, drugstopoffset, sedative FROM public.pivoted_sedative
        WHERE patientunitstayid={patientunitstayid} 
        AND chartoffset <= '{endoffset}'::decimal AND drugstopoffset >= '{startoffset}'::decimal
        AND sedative=1
        """

    sedativeChart = pd.read_sql_query(query.format(patientunitstayid=patientunitstayid, startoffset=startoffset, endoffset=endoffset), con=con)
    timeSeries['sedative medications'] = pd.Series(0 if sedativeChart.empty else 1 for time in timeSeries['Time'])

def addVentilation(patientunitstayid, timeSeries, con):
    startoffset = timeSeries['chartoffset'][0]
    endoffset = timeSeries['chartoffset'][len(timeSeries) - 1]

    query = """
            SELECT * FROM public.pivoted_venti
            WHERE patientunitstayid={patientunitstayid}
            AND ((respcarestatusoffset>={startoffset} AND respcarestatusoffset<={endoffset}) 
            OR (respchartoffset>={startoffset} AND respchartoffset<={endoffset})
            OR (treatmentoffset>={startoffset} AND treatmentoffset<={endoffset}))
            """

    ventChart = pd.read_sql_query(query.format(patientunitstayid=patientunitstayid, startoffset=startoffset, endoffset=endoffset), con=con)
    timeSeries['ventilation'] = pd.Series(0 if ventChart.empty else 1 for time in timeSeries['Time'])