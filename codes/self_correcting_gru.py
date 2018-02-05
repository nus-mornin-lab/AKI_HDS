import tensorflow as tf
from data_iterator import DataIterator


numFeatures = 24
batchSize = 256
stateSizes = (16, 8)


def buildGraph(numFeatures=numFeatures, stateSizes=stateSizes):
    x = tf.placeholder(tf.float64, [None, None, numFeatures])  # [batchSize, num_steps, numFeatures]
    batchSize = tf.shape(x)[0]
    num_steps = tf.shape(x)[1]
    keepProb = tf.placeholder(tf.float64, shape=())
    x_dropout = tf.nn.dropout(x, keepProb)
    x_transpose = tf.transpose(x_dropout, [1, 0, 2])  # [num_steps, batchSize, numFeatures]
    x_ta = tf.TensorArray(dtype=tf.float64, size=num_steps, name='x_ta').unstack(x_transpose)
    urineOutput = tf.placeholder(tf.float64, [None, None])  # [batchSize, num_steps]
    urineOutput_transpose = tf.transpose(urineOutput, [1, 0])  # [num_steps, batchSize]
    urineOutput_ta = tf.TensorArray(dtype=tf.float64, size=num_steps, name='urineOutput_ta').unstack(
        urineOutput_transpose)
    sequenceLengths = tf.placeholder(tf.int32, [None])  # [batchSize]
    y = tf.placeholder(tf.float64, [None, None])  # [batchSize, num_steps]
    y_transpose = tf.transpose(y, [1, 0])  # [num_steps, batchSize]
    y_ta = tf.TensorArray(dtype=tf.float64, size=num_steps,  name='y_ta').unstack(y_transpose)
    y_predicted_ta = tf.TensorArray(dtype=tf.float64, size=num_steps, name='y_predicted_ta', clear_after_read=False)
    mask = tf.placeholder(tf.float64, [None, None])  # [batchSize, num_steps]

    def getCell(stateSize):
        cell = tf.nn.rnn_cell.GRUCell(stateSize, activation=tf.nn.relu)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keepProb)
        return cell

    rnnLayers = [getCell(stateSize) for stateSize in stateSizes]
    feedback_place_holder = tf.get_variable('feedback_place_holder', [1, 16], dtype=tf.float64)
    feedback_place_holder_tiled = tf.tile(feedback_place_holder, [batchSize, 1])
    W = tf.get_variable('W', [stateSizes[-1], 1], initializer=tf.contrib.layers.xavier_initializer(),
                        dtype=tf.float64)
    b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0), dtype=tf.float64)
    feedback_W1 = tf.get_variable('feedback_W1', [3, 16], initializer=tf.contrib.layers.xavier_initializer(),
                        dtype=tf.float64)
    feedback_b1 = tf.get_variable('feedback_b1', [16], initializer=tf.constant_initializer(0.0), dtype=tf.float64)
    feedback_W2 = tf.get_variable('feedback_W2', [16, 16], initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float64)
    feedback_b2 = tf.get_variable('feedback_b2', [16], initializer=tf.constant_initializer(0.0), dtype=tf.float64)

    def condition(unused_time, elements_finished, *_):
        return tf.logical_not(tf.reduce_all(elements_finished))

    def body(time, elements_finished, current_input, states, y_predicted_ta, urineOutput_ta, y_ta):
        """Internal while loop body for raw_rnn.
        Args:
            time: time scalar.
            elements_finished: batch-size vector.
            current_input: possibly nested tuple of input tensors.
            emit_ta: possibly nested tuple of output TensorArrays.
            state: possibly nested tuple of state tensors.
            loop_state: possibly nested tuple of loop state tensors.
        Returns:
            Tuple having the same size as Args but with updated values.
        """
        def body_without_feedback(current_input, states, y_predicted_ta, urineOutput_ta, y_ta):
            new_states = [None]*len(states)
            last_output = current_input
            for i, cell in enumerate(rnnLayers):
                # set scope name to avoid duplicated variable names in rnn cells
                with tf.variable_scope('rnn_layer'+str(i+1)) as scope:
                    state = states[i]
                    last_output = tf.concat([last_output, feedback_place_holder_tiled], axis=1)
                    (last_output, cell_state) = cell(last_output, state)
                    new_states[i] = cell_state
            logits = tf.matmul(last_output, W) + b  #[batchSize, 1]
            predicts = tf.nn.sigmoid(logits)
            return predicts, new_states, y_predicted_ta, urineOutput_ta, y_ta

        def body_with_feedback(current_input, states, y_predicted_ta, urineOutput_ta, y_ta):
            new_states = [None] * len(states)
            last_output = current_input
            y_predicted_feedback = y_predicted_ta.read(time - 6)  # [batchSize]
            y_feedback = y_ta.read(time - 6)  # [batchSize]
            urineOutput_feedback = urineOutput_ta.read(time)  # [batchSize]
            y_predicted_feedback = tf.expand_dims(y_predicted_feedback, -1)
            y_feedback = tf.expand_dims(y_feedback, -1)
            urineOutput_feedback = tf.expand_dims(urineOutput_feedback, -1)
            feedback = tf.concat([y_predicted_feedback, y_feedback, urineOutput_feedback], axis=1)  # [batchSize, 3]
            feedback = tf.matmul(feedback, feedback_W1) + feedback_b1
            feedback = tf.nn.relu(feedback)
            feedback = tf.matmul(feedback, feedback_W2) + feedback_b2  # [batchSize, 16]
            feedback = tf.nn.relu(feedback)
            for i, cell in enumerate(rnnLayers):
                # set scope name to avoid duplicated variable names in rnn cells
                with tf.variable_scope('rnn_layer' + str(i + 1)) as scope:
                    state = states[i]
                    last_output = tf.concat([last_output, feedback], axis=1)
                    (last_output, cell_state) = cell(last_output, state)
                    new_states[i] = cell_state
            logits = tf.matmul(last_output, W) + b  # [batchSize, 1]
            predicts = tf.nn.sigmoid(logits)
            return predicts, new_states, y_predicted_ta, urineOutput_ta, y_ta

        predicts, new_states, y_predicted_ta, urineOutput_ta, y_ta = tf.cond(time > 6,
                                                                             lambda: body_with_feedback(current_input,
                                                                                                        states,
                                                                                                        y_predicted_ta,
                                                                                                        urineOutput_ta,
                                                                                                        y_ta),
                                                                             lambda: body_without_feedback(
                                                                                 current_input, states, y_predicted_ta,
                                                                                 urineOutput_ta, y_ta))
        predicts = tf.reshape(predicts, [batchSize])
        y_predicted_ta = y_predicted_ta.write(time, predicts)
        elements_finished = (time >= (sequenceLengths-1))
        time += 1
        next_input = tf.cond(tf.reduce_all(elements_finished), lambda: current_input, lambda: x_ta.read(time))
        return time, elements_finished, next_input, new_states, y_predicted_ta, urineOutput_ta, y_ta

    # init loop params
    time = tf.Variable(0)
    elements_finished = (time >= (sequenceLengths-1))
    first_input = x_ta.read(0)
    states = [cell.zero_state(batchSize, tf.float64) for cell in rnnLayers]
    returned = tf.while_loop(condition, body,
                  loop_vars=[time, elements_finished, first_input, states, y_predicted_ta, urineOutput_ta, y_ta])
    y_predicted_ta = returned[-3]

    # get result from tensor array
    y_predicted = y_predicted_ta.stack()  # [num_steps, batchSize]
    y_predicted = tf.transpose(y_predicted, [1, 0])  # [batchSize, num_steps]
    costs = -y * tf.log(tf.clip_by_value(y_predicted, 1e-10, 1.0)) - (1 - y) * tf.log(
        tf.clip_by_value(1 - y_predicted, 1e-10, 1.0))
    costs *= mask
    cost = tf.reduce_mean(tf.reduce_sum(costs, reduction_indices=[1]))
    accuracy = tf.reduce_sum(tf.cast(tf.abs(y_predicted - y) < 0.5, tf.float64) * mask) / tf.reduce_sum(mask)
    learningRate = tf.placeholder(tf.float64, shape=())
    momentum = tf.placeholder(tf.float64, shape=())
    trainStepAdam = tf.train.AdamOptimizer(learningRate).minimize(cost)
    trainStepMomentum = tf.train.MomentumOptimizer(learningRate, momentum).minimize(cost)
    return {
        'x': x,
        'urineOutput': urineOutput,
        'y': y,
        'seqlen': sequenceLengths,
        'mask': mask,
        'adam': trainStepAdam,
        'momentum': trainStepMomentum,
        'momentumValue': momentum,
        'predicts': y_predicted,
        'accuracy': accuracy,
        'cost': cost,
        'keepProb': keepProb,
        'learningRate': learningRate,
        'feedback_place_holder': feedback_place_holder,
        'feedback_W1': feedback_W1,
        'feedback_W2': feedback_W2,
        'rnn': [v for v in tf.global_variables() if v.name.startswith('rnn_layer')]
    }


def trainGraph(g, sess, train, test, epochs=10, batchSize=batchSize, learningRate=1e-4, momentum=0.9, optimizer='adam'):
    tr = DataIterator(train)
    te = DataIterator(test)

    step, accuracy = 0, 0
    trLosses = []
    current_epoch = 0
    totalCost = 0
    while current_epoch < epochs:
        step += 1
        batch = tr.next_batch(batchSize)
        feed = {g['x']: batch[0][:, :, :-1], g['urineOutput']: batch[0][:, :, -1], g['y']: batch[1],
                g['seqlen']: batch[2], g['mask']: batch[3], g['keepProb']: 0.5, g['learningRate']: learningRate}
        if optimizer == 'momentum':
            feed[g['momentumValue']] = momentum
        accuracy_, cost, _ = sess.run([g['accuracy'], g['cost'], g[optimizer]], feed_dict=feed)
        totalCost += cost
        accuracy += accuracy_

        if tr.epochs > current_epoch:
            current_epoch += 1
            trLosses.append(accuracy / step)
            # eval test set
            test_epoch = te.epochs
            testAccuracy = 0
            while te.epochs <= test_epoch:
                batch = te.next_batch(batchSize)
                feed = {g['x']: batch[0][:, :, :-1], g['urineOutput']: batch[0][:, :, -1], g['y']: batch[1],
                        g['seqlen']: batch[2], g['mask']: batch[3], g['keepProb']: 1}
                size = len(batch[0])
                testAccuracy += sess.run([g['accuracy']], feed_dict=feed)[0]*size
            print("Accuracy after epoch", current_epoch, "cost: ", totalCost / step, " - tr:", trLosses[-1], "- te:",
                  testAccuracy / te.size)
            step, accuracy, totalCost = 0, 0, 0
