import tensorflow as tf

numFeatures = 25


def buildGraph(numFeatures=numFeatures, alpha=0.01):
    x = tf.placeholder(tf.float64, [None, numFeatures])
    y = tf.placeholder(tf.float64, [None])
    mask = tf.placeholder(tf.float64, [None])
    W = tf.get_variable('W', [numFeatures, 1], initializer=tf.random_normal_initializer(),
                        dtype=tf.float64)
    b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0), dtype=tf.float64)
    predicts = tf.sigmoid(tf.matmul(x, W) + b)
    predicts = tf.reshape(predicts, [-1])
    costs = -y * tf.log(tf.clip_by_value(predicts, 1e-10, 1.0)) - (1 - y) * tf.log(
        tf.clip_by_value(1 - predicts, 1e-10, 1.0))
    costs *= mask
    cost = tf.reduce_mean(costs) + tf.reduce_sum(tf.abs(W)) * alpha
    accuracy = tf.reduce_sum(tf.cast(tf.abs(predicts - y) < 0.5, tf.float64) * mask) / tf.reduce_sum(mask)
    trainStep = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
    return {
        'x': x,
        'y': y,
        'mask': mask,
        'trainStep': trainStep,
        'predicts': predicts,
        'accuracy': accuracy,
        'cost': cost,
        'W': W
    }

def trainGraph(g, sess, train_X, train_y, train_mask, test_X, test_y, test_mask, batchSize=2048, epoch=50):
    for i in range(epoch):
        index = 0
        accuracy, cost = 0, 0
        while index < len(train_X):
            end = min(index+batchSize, len(train_X))
            feed = {g['x']: train_X[index:end], g['y']: train_y[index:end], g['mask']: train_mask[index:end]}
            accuracy_, cost_, _ = sess.run([g['accuracy'], g['cost'], g['trainStep']], feed_dict=feed)
            size = end-index
            accuracy += size*accuracy_
            cost += size*cost_
            index = end
        cost /= len(train_X)
        accuracy /= len(train_X)
        test_index = 0
        test_accuracy = 0
        while test_index < len(test_X):
            test_end = min(test_index + batchSize, len(test_X))
            feed = {g['x']: test_X[test_index:test_end], g['y']: test_y[test_index:test_end],
                    g['mask']: test_mask[test_index:test_end]}
            test_accuracy_ = sess.run(g['accuracy'], feed_dict=feed)
            size = test_end - test_index
            test_accuracy += size * test_accuracy_
            test_index = test_end
        test_accuracy /= len(test_X)
        print("Accuracy after epoch", i+1, "cost: ", cost, " - tr:", accuracy, "- te:", test_accuracy)

