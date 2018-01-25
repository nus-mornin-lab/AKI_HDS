import tensorflow as tf
from data_iterator import DataIterator

numFeatures = 24
batchSize = 256
stateSizes = (256, 128, 64, 32)
keepProb = 0.5


def buildGraph(numFeatures=numFeatures, batchSize=batchSize, stateSizes=stateSizes, keepProb=keepProb):
    x = tf.placeholder(tf.float64, [batchSize, None, numFeatures])  # [batchSize, num_steps, numFeatures]
    sequenceLengths = tf.placeholder(tf.int32, [batchSize])  # [batchSize]
    y = tf.placeholder(tf.float64, [batchSize, None])  # [batchSize, num_steps]
    musk = tf.placeholder(tf.float64, [batchSize, None])  # [batchSize, num_steps]

    def getCell(stateSize):
        cell = tf.nn.rnn_cell.LSTMCell(stateSize, forget_bias=3.0, activation=tf.nn.relu, state_is_tuple=True)
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
    losses = -y * tf.log(tf.clip_by_value(predicts, 1e-10, 1.0)) - (1 - y) * tf.log(
        tf.clip_by_value(1 - predicts, 1e-10, 1.0))
    losses *= musk
    loss = tf.reduce_mean(tf.reduce_sum(losses, reduction_indices=[1]))
    accuracy = tf.reduce_sum(tf.cast(tf.abs(predicts - y) < 0.5, tf.float64) * musk) / tf.reduce_sum(musk)
    trainStep = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return {
        'x': x,
        'y': y,
        'seqlen': sequenceLengths,
        'musk': musk,
        'trainStep': trainStep,
        'predicts': predicts,
        'accuracy': accuracy
    }


def trainGraph(g, sess, train, test, epochs=10):
    tr = DataIterator(train)
    te = DataIterator(test)

    step, accuracy = 0, 0
    trLosses, teLosses = [], []
    current_epoch = 0
    while current_epoch < epochs:
        step += 1
        batch = tr.next_batch(batchSize)
        feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['musk']: batch[3]}
        accuracy_, _ = sess.run([g['accuracy'], g['trainStep']], feed_dict=feed)
        accuracy += accuracy_

        if tr.epochs > current_epoch:
            current_epoch += 1
            trLosses.append(accuracy / step)
            step, accuracy = 0, 0

            # eval test set
            teEpoch = te.epochs
            while te.epochs == teEpoch:
                step += 1
                batch = te.next_batch(batchSize)
                feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['musk']: batch[3]}
                accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
                accuracy += accuracy_

            teLosses.append(accuracy / step)
            step, accuracy = 0, 0
            print("Accuracy after epoch", current_epoch, " - tr:", trLosses[-1], "- te:", teLosses[-1])
