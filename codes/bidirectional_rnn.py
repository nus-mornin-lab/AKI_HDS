import  tensorflow as tf
from tensorflow.contrib import rnn
from data_iterator import DataIterator


numFeatures = 29
batchSize = 64
stateSize = 64

def buildGraph():
	x = tf.placeholder(tf.float64, [batchSize, None, numFeatures])  # [batchSize, num_steps, numFeatures]
	sequenceLengths = tf.placeholder(tf.int32, [batchSize]) # [batchSize]
	y = tf.placeholder(tf.int32, [batchSize, None]) # [batchSize, num_steps]
	musk = tf.placeholder(tf.int32, [batchSize, None]) # [batchSize, num_steps]
	# Forward direction cell
	lstmFwCell = rnn.BasicLSTMCell(stateSize, forget_bias=1.0)
	# Backward direction cell
	lstmBwCell = rnn.BasicLSTMCell(stateSize, forget_bias=1.0)
	outputs, states = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, x, sequenceLengths)
	with tf.variable_scope('softmax'):
		W = tf.get_variable('W', [stateSize], initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))
	logits = tf.matmul(outputs, W) + b # [batchSize, num_steps]
	predicts = tf.nn.sigmoid(logits)
	losses = -y*tf.log(predicts) - (1-y)*tf.log(1-predicts)
	losses *= musk
	loss = tf.reduce_mean(tf.reduce_sum(losses, reduction_indices=[1]))
	accuracy = tf.reduce_sum(tf.abs(predicts-y)*musk < 0.5)/tf.reduce_sum(musk)
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


def trainGraph(g, train, test, epochs = 10):
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		tr = DataIterator(train)
		te = DataIterator(test)

		step, accuracy = 0, 0
		trLosses, teLosses = [], []
		current_epoch = 0
		while current_epoch < epochs:
			step += 1
			batch = tr.next_batch(batchSize)
			feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
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
					feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
					accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
					accuracy += accuracy_

				teLosses.append(accuracy / step)
				step, accuracy = 0, 0
				print("Accuracy after epoch", current_epoch, " - tr:", trLosses[-1], "- te:", teLosses[-1])
