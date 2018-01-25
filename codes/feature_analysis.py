from connection import *
import json


engine = getEngine()
features_json_data = open('./features.json').read()
features = json.loads(features_json_data)
print()

for feature in features:
    for table in ('chartevents', 'labevents'):
        if table not in feature:
            continue
        itemIDs = str(tuple(feature[table])) if len(feature[table]) > 1 else '(' + str(feature[table][0]) + ')'
        print("{label} in {table}:".format(label=feature['label'], table=table))
        # result = pd.read_sql("""
        # SELECT MAX(valuenum), MIN(valuenum), AVG(valuenum), STDDEV(valuenum)
        # FROM {table} WHERE itemid IN {itemIDs};
        # """.format(table=table, itemIDs=itemIDs), engine).iloc[0]
        # print("Max: {max}, Min: {min}, Average: {avg}, Standard deviation: {stddev}".format(
        #     max=result['max'], min=result['min'], avg=result['avg'], stddev=result['stddev']))
        result = pd.read_sql("""
                SELECT percentile_disc(0.1) WITHIN GROUP (ORDER BY valuenum) AS percentile_1,
                percentile_disc(0.01) WITHIN GROUP (ORDER BY valuenum) AS percentile_01,
                percentile_disc(0.005) WITHIN GROUP (ORDER BY valuenum) AS percentile_005,
                percentile_disc(0.001) WITHIN GROUP (ORDER BY valuenum) AS percentile_001,
                percentile_disc(0.0005) WITHIN GROUP (ORDER BY valuenum) AS percentile_0005,
                percentile_disc(0.0001) WITHIN GROUP (ORDER BY valuenum) AS percentile_0001,
                percentile_disc(0.00005) WITHIN GROUP (ORDER BY valuenum) AS percentile_00005,
                percentile_disc(0.00001) WITHIN GROUP (ORDER BY valuenum) AS percentile_00001,
                percentile_disc(0.9) WITHIN GROUP (ORDER BY valuenum) AS percentile_9,
                percentile_disc(0.99) WITHIN GROUP (ORDER BY valuenum) AS percentile_99,
                percentile_disc(0.999) WITHIN GROUP (ORDER BY valuenum) AS percentile_999,
                percentile_disc(0.9995) WITHIN GROUP (ORDER BY valuenum) AS percentile_9995,
                percentile_disc(0.9999) WITHIN GROUP (ORDER BY valuenum) AS percentile_9999,
                percentile_disc(0.99995) WITHIN GROUP (ORDER BY valuenum) AS percentile_99995,
                percentile_disc(0.99999) WITHIN GROUP (ORDER BY valuenum) AS percentile_99999,
                percentile_disc(0.5) WITHIN GROUP (ORDER BY valuenum) AS median
                FROM {table} WHERE itemid IN {itemIDs}; 
                """.format(table=table, itemIDs=itemIDs), engine)
        print(result.iloc[0])
        print("\n\n")
