import tensorflow as tf
from data_iterator import DataIterator


numFeatures = 24
batchSize = 256
stateSizes = (16, 8)


def buildGraph(numFeatures=numFeatures, stateSizes=stateSizes):
    x = tf.placeholder(tf.float32, [None, None, numFeatures])  # [batchSize, num_steps, numFeatures]
    batchSize = tf.shape(x)[0]
    num_steps = tf.shape(x)[1]
    x_shifted = tf.slice(x, [0, 1, 0], [batchSize, num_steps-1, numFeatures])
    x_predicted_ta = tf.TensorArray(dtype=tf.float32, size=num_steps, name='x_predicted_ta', clear_after_read=False)
    keepProb = tf.placeholder(tf.float32, shape=())
    x_transpose = tf.transpose(x, [1, 0, 2])  # [num_steps, batchSize, numFeatures]
    x_ta = tf.TensorArray(dtype=tf.float32, size=num_steps, name='x_ta').unstack(x_transpose)
    sequenceLengths = tf.placeholder(tf.int32, [None])  # [batchSize]
    y = tf.placeholder(tf.float32, [None, None])  # [batchSize, num_steps]
    y_transpose = tf.transpose(y, [1, 0])  # [num_steps, batchSize]
    y_ta = tf.TensorArray(dtype=tf.float32, size=num_steps,  name='y_ta').unstack(y_transpose)
    y_predicted_ta = tf.TensorArray(dtype=tf.float32, size=num_steps, name='y_predicted_ta', clear_after_read=False)
    mask = tf.placeholder(tf.float32, [None, None])  # [batchSize, num_steps]

    def getCell(stateSize):
        cell = tf.nn.rnn_cell.GRUCell(stateSize, activation=tf.nn.relu)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keepProb)
        return cell

    rnnLayers = [getCell(stateSize) for stateSize in stateSizes]
    feedback_place_holder = tf.get_variable('feedback_place_holder', [1, 16], dtype=tf.float32)
    feedback_place_holder_tiled = tf.tile(feedback_place_holder, [batchSize, 1])
    W = tf.get_variable('W', [stateSizes[-1], 1], initializer=tf.contrib.layers.xavier_initializer(),
                        dtype=tf.float32)
    b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    W_regularizer = tf.get_variable('W_regularizer', [stateSizes[-1], numFeatures],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    b_regularizer = tf.get_variable('b_regularizer', [numFeatures], initializer=tf.constant_initializer(0.0),
                                    dtype=tf.float32)
    feedback_W1 = tf.get_variable('feedback_W1', [2 + 2 * numFeatures, 16],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float32)
    feedback_b1 = tf.get_variable('feedback_b1', [16], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    feedback_W2 = tf.get_variable('feedback_W2', [16, 16], initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float32)
    feedback_b2 = tf.get_variable('feedback_b2', [16], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

    def condition(unused_time, elements_finished, *_):
        return tf.logical_not(tf.reduce_all(elements_finished))

    def body(time, elements_finished, current_input, states, y_predicted_ta, y_ta, x_predicted_ta):
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
        def body_without_feedback(current_input, states, y_predicted_ta, y_ta):
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
            x_predicted = tf.sigmoid(tf.matmul(last_output, W_regularizer) + b_regularizer)  # [batchSize, numFeatures]
            return predicts, x_predicted, new_states, y_predicted_ta, y_ta

        def body_with_feedback(current_input, states, y_predicted_ta, y_ta, x_predicted_ta):
            new_states = [None] * len(states)
            last_output = current_input
            y_predicted_feedback = y_predicted_ta.read(time - 6)  # [batchSize]
            y_feedback = y_ta.read(time - 6)  # [batchSize]
            y_predicted_feedback = tf.expand_dims(y_predicted_feedback, -1)
            y_feedback = tf.expand_dims(y_feedback, -1)
            x_predicted_feedback = x_predicted_ta.read(time-1)
            feedback = tf.concat(
                [y_predicted_feedback, y_feedback, x_predicted_feedback,
                 current_input],
                axis=1)  # [batchSize, 2+2*numFeatures]
            feedback = tf.stop_gradient(feedback)
            feedback = tf.matmul(feedback, feedback_W1) + feedback_b1
            feedback = tf.nn.relu(feedback)
            feedback = tf.matmul(feedback, feedback_W2) + feedback_b2  # [batchSize, 256]
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
            x_predicted = tf.sigmoid(tf.matmul(last_output, W_regularizer) + b_regularizer)  # [batchSize, numFeatures]
            return predicts, x_predicted, new_states, y_predicted_ta, y_ta

        predicts, x_predicted, new_states, y_predicted_ta, y_ta = tf.cond(time > 6,
                                                                                          lambda: body_with_feedback(
                                                                                              current_input,
                                                                                              states,
                                                                                              y_predicted_ta,
                                                                                              y_ta, x_predicted_ta),
                                                                                          lambda: body_without_feedback(
                                                                                              current_input, states,
                                                                                              y_predicted_ta, y_ta))
        predicts = tf.reshape(predicts, [batchSize])
        y_predicted_ta = y_predicted_ta.write(time, predicts)
        x_predicted_ta = x_predicted_ta.write(time, x_predicted)
        elements_finished = (time >= (sequenceLengths-1))
        time += 1
        next_input = tf.cond(tf.reduce_all(elements_finished), lambda: current_input, lambda: x_ta.read(time))
        return time, elements_finished, next_input, new_states, y_predicted_ta, y_ta, x_predicted_ta

    # init loop params
    time = tf.Variable(0)
    elements_finished = (time >= (sequenceLengths-1))
    first_input = x_ta.read(0)
    states = [cell.zero_state(batchSize, tf.float32) for cell in rnnLayers]
    returned = tf.while_loop(condition, body,
                             loop_vars=[time, elements_finished, first_input, states, y_predicted_ta,
                                        y_ta, x_predicted_ta])
    y_predicted_ta = returned[-3]
    x_predicted_ta = returned[-1]

    # get result from tensor array
    y_predicted = y_predicted_ta.stack()  # [num_steps, batchSize]
    y_predicted = tf.transpose(y_predicted, [1, 0])  # [batchSize, num_steps]
    x_predicted = x_predicted_ta.stack()  # [num_steps, batchSize, numFeatures]
    x_predicted = tf.transpose(x_predicted, [1, 0, 2])  # [batchSize, num_steps, numFeatures]
    x_predicted = tf.slice(x_predicted, [0, 0, 0], [batchSize, num_steps-1, numFeatures])
    mask_sliced = tf.slice(mask, [0, 1], [batchSize, num_steps-1])
    mask_sliced = tf.expand_dims(mask_sliced, -1)
    mask_sliced = tf.tile(mask_sliced, [1, 1, numFeatures])
    regularizer = ((x_predicted - x_shifted) ** 2) * mask_sliced
    regularizer = tf.reduce_mean(tf.reduce_sum(regularizer, reduction_indices=[1]))
    regularizer_weight = tf.placeholder_with_default(tf.cast(0.05, tf.float32), shape=())
    objective_costs = -y * tf.log(tf.clip_by_value(y_predicted, 1e-10, 1.0-1e-10)) - (1 - y) * tf.log(
        tf.clip_by_value(1 - y_predicted, 1e-10, 1.0-1e-10))
    objective_costs *= mask
    objective_cost = tf.reduce_mean(tf.reduce_sum(objective_costs, reduction_indices=[1]))
    cost = objective_cost + regularizer * regularizer_weight
    accuracy = tf.reduce_sum(tf.cast(tf.abs(y_predicted - y) < 0.5, tf.float32) * mask) / tf.reduce_sum(mask)
    learningRate = tf.placeholder(tf.float32, shape=())
    momentum = tf.placeholder(tf.float32, shape=())
    trainStepAdam = tf.train.AdamOptimizer(learningRate).minimize(cost)
    trainStepMomentum = tf.train.MomentumOptimizer(learningRate, momentum).minimize(cost)
    return {
        'x': x,
        'y': y,
        'seqlen': sequenceLengths,
        'mask': mask,
        'adam': trainStepAdam,
        'momentum': trainStepMomentum,
        'momentumValue': momentum,
        'predicts': y_predicted,
        'accuracy': accuracy,
        'cost': cost,
        'objective_cost': objective_cost,
        'regularization_cost': regularizer,
        'keepProb': keepProb,
        'learningRate': learningRate,
        'regularizer_weight': regularizer_weight,
        'feedback_place_holder': feedback_place_holder,
        'feedback_W1': feedback_W1,
        'feedback_W2': feedback_W2,
        'rnn': [v for v in tf.global_variables() if v.name.startswith('rnn_layer')]
    }


def trainGraph(g, sess, train, test, epochs=10, batchSize=batchSize, learningRate=1e-4, momentum=0.9, optimizer='adam', regularizer_weight=0.05):
    tr = DataIterator(train)
    te = DataIterator(test)

    step, accuracy = 0, 0
    trLosses = []
    current_epoch = 0
    totalCost = 0
    totalObjectiveCost = 0
    totalRegularizationCost = 0
    while current_epoch < epochs:
        step += 1
        batch = tr.next_batch(batchSize)
        feed = {g['x']: batch[0],
                g['y']: batch[1],
                g['seqlen']: batch[2],
                g['mask']: batch[3],
                g['keepProb']: 0.5,
                g['learningRate']: learningRate,
                g['regularizer_weight']: regularizer_weight
                }
        if optimizer == 'momentum':
            feed[g['momentumValue']] = momentum
        accuracy_, cost, objective_cost, regularization_cost, _ = sess.run(
            [g['accuracy'], g['cost'], g['objective_cost'], g['regularization_cost'], g[optimizer]], feed_dict=feed)
        totalCost += cost
        totalObjectiveCost += objective_cost
        totalRegularizationCost += regularization_cost
        accuracy += accuracy_

        if tr.epochs > current_epoch:
            current_epoch += 1
            trLosses.append(accuracy / step)
            # eval test set
            test_epoch = te.epochs
            testAccuracy = 0
            while te.epochs <= test_epoch:
                batch = te.next_batch(batchSize)
                feed = {g['x']: batch[0], g['y']: batch[1],
                        g['seqlen']: batch[2], g['mask']: batch[3], g['keepProb']: 1}
                size = len(batch[0])
                testAccuracy += sess.run([g['accuracy']], feed_dict=feed)[0]*size
            print("Accuracy after epoch", current_epoch, "cost: ", totalCost / step, "objective cost: ",
                  totalObjectiveCost / step, "regularization cost: ", totalRegularizationCost / step, " - tr:",
                  trLosses[-1], "- te:",
                  testAccuracy / te.size)
            step, accuracy, totalCost = 0, 0, 0
            totalObjectiveCost, totalRegularizationCost = 0, 0
