import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import  numpy as np
import sys
from data_extraction import normalizeFeatures
from data_iterator import DataIterator

modelName = sys.argv[2]
modelName = modelName + '/' + modelName + '.ckpt'
stateSizes = sys.argv[3:]
stateSizes = [int(size) for size in stateSizes]
modelType = sys.argv[1]
# build the same graph
tf.reset_default_graph()
if modelType == 'correcting':
    import self_correcting_gru
    graph = self_correcting_gru.buildGraph(stateSizes=stateSizes)
else:
    import multi_layer_lstm
    graph = multi_layer_lstm.buildGraph(stateSizes=stateSizes)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, modelName)
    print("Model restored. Loading data.")
    allTimeSeries = normalizeFeatures(None)
    columns = ['age', 'gender', 'Heart Rate',
               'Respiratory Rate', 'SpO2/SaO2', 'pH', 'Potassium', 'Calcium',
               'Glucose', 'Sodium', 'HCO3', 'White Blood Cells', 'Hemoglobin',
               'Red Blood Cells', 'Platelet Count', 'Weight', 'Urea Nitrogen',
               'Creatinine', 'Blood Pressure', 'gcs',
               'ventilation', 'vasoactive medications', 'sedative medications', '1 hours urine output',
               '6 hours urine output', 'AKI']
    allTimeSeries = [timeSeries[columns] for timeSeries in allTimeSeries]
    _, test = train_test_split(allTimeSeries, test_size=0.2, random_state=12)
    te = DataIterator(test)
    batch = te.next_batch(len(test))
    print("Data loaded.")
    y = batch[1]
    seqlen = batch[2]
    if modelType == 'correcting':
        feed = {graph['x']: batch[0][:, :, :-1], graph['urineOutput']: batch[0][:, :, -1], graph['y']: batch[1],
                graph['seqlen']: batch[2], graph['mask']: batch[3], graph['keepProb']: 1}
    else:
        feed = {graph['x']: batch[0], graph['y']: batch[1], graph['seqlen']: batch[2], graph['mask']: batch[3], graph['keepProb']: 1}
    predicts = sess.run(graph['predicts'], feed_dict=feed)
    predicts_with_seqlen = []
    y_with_seqlen = []
    for i in range(len(test)):
        predicts_with_seqlen.append(predicts[i, :seqlen[i]])
        y_with_seqlen.append(y[i, :seqlen[i]])
    predicts = np.hstack(predicts_with_seqlen)
    y = np.hstack(y_with_seqlen)
    print("AUC is ", roc_auc_score(y, predicts))
    predicts = (predicts >= 0.5)
    isCorrect = (predicts == y)
    print("Accuracy: ", isCorrect.sum()/len(isCorrect))
