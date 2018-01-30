import tensorflow as tf
from data_iterator import DataIterator

numFeatures = 24
batchSize = 256
stateSizes = (256, 128, 64, 32)
keepProb = 0.5


def buildGraph(numFeatures=numFeatures, stateSizes=stateSizes, keepProb=keepProb):
    x = tf.placeholder(tf.float64, [None, None, numFeatures])  # [batchSize, num_steps, numFeatures]
    sequenceLengths = tf.placeholder(tf.int32, [None])  # [batchSize]
    y = tf.placeholder(tf.float64, [None, None])  # [batchSize, num_steps]
    musk = tf.placeholder(tf.float64, [None, None])  # [batchSize, num_steps]
    batchSize = tf.shape(x)[0]

    def getCell(stateSize):
        cell = tf.nn.rnn_cell.GRUCell(stateSize, activation=tf.nn.relu)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keepProb)
        return cell

    rnnLayers = [getCell(stateSize) for stateSize in stateSizes]
    multiLSTM = tf.nn.rnn_cell.MultiRNNCell(rnnLayers)
    outputs, states = tf.nn.dynamic_rnn(multiLSTM, x, sequenceLengths, dtype=tf.float64)
    outputs = tf.reshape(outputs, [-1, stateSizes[-1]])
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [stateSizes[-1], 1], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0), dtype=tf.float64)
    logits = tf.matmul(outputs, W) + b  # [batchSize, num_steps]
    predicts = tf.nn.sigmoid(logits)
    predicts = tf.reshape(predicts, [batchSize, -1])
    costs = -y * tf.log(tf.clip_by_value(predicts, 1e-10, 1.0)) - (1 - y) * tf.log(
        tf.clip_by_value(1 - predicts, 1e-10, 1.0))
    costs *= musk
    cost = tf.reduce_mean(tf.reduce_sum(costs, reduction_indices=[1]))
    accuracy = tf.reduce_sum(tf.cast(tf.abs(predicts - y) < 0.5, tf.float64) * musk) / tf.reduce_sum(musk)
    trainStep = tf.train.AdamOptimizer(1e-4).minimize(cost)
    return {
        'x': x,
        'y': y,
        'seqlen': sequenceLengths,
        'musk': musk,
        'trainStep': trainStep,
        'predicts': predicts,
        'accuracy': accuracy,
        'cost': cost
    }


def trainGraph(g, sess, train, test, epochs=10, batchSize=batchSize):
    tr = DataIterator(train)
    te = DataIterator(test)

    step, accuracy = 0, 0
    trLosses = []
    current_epoch = 0
    totalCost = 0
    while current_epoch < epochs:
        step += 1
        batch = tr.next_batch(batchSize)
        feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['musk']: batch[3]}
        accuracy_, cost, _ = sess.run([g['accuracy'], g['cost'], g['trainStep']], feed_dict=feed)
        totalCost += cost
        accuracy += accuracy_

        if tr.epochs > current_epoch:
            current_epoch += 1
            trLosses.append(accuracy / step)
            # eval test set
            batch = te.next_batch()
            feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['musk']: batch[3]}
            accuracy = sess.run([g['accuracy']], feed_dict=feed)[0]
            print("Accuracy after epoch", current_epoch, "cost: ", totalCost/step, " - tr:", trLosses[-1], "- te:", accuracy)
            step, accuracy, totalCost = 0, 0, 0
