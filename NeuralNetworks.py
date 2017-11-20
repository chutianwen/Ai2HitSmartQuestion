import tensorflow as tf
import os
import numpy as np
from AppUtils import logger, TaskReporter
from collections import defaultdict

class NeuralNetworks:
	def __init__(self, config_parser):
		"""
        features, targets, split_fraction=0.8,
                 embed_size=300, lstm_size=256, lstm_layers=1, batch_size=500, learning_rate=0.001,
                 keep_prob=0.5, epochs=15
        """
		self.model_name = config_parser['Path']['model_name']
		self.save_model_path = config_parser['Path']['save_model_path']
		self.embed_size = int(config_parser['ModelParas']['embed_size'])
		self.lstm_size = int(config_parser['ModelParas']['lstm_size'])
		self.lstm_layer = int(config_parser['ModelParas']['lstm_layer'])
		self.batch_size = int(config_parser['ModelParas']['batch_size'])
		self.learning_rate = float(config_parser['ModelParas']['learning_rate'])
		self.keep_prob = float(config_parser['ModelParas']['keep_prob'])
		self.epochs = int(config_parser['ModelParas']['epochs'])
		self.checkpoint_time = int(config_parser['ModelParas']['checkpoint_time'])
		self.validation_time = int(config_parser['ModelParas']['validation_time'])

		path_vocab_to_int = config_parser['Path']['path_vocab_to_int']
		if os.path.exists(path_vocab_to_int):
			self.vocab_to_int = np.load(path_vocab_to_int).item()
			assert isinstance(self.vocab_to_int, dict), "!vocab_to_int is not python dict type"
			self.vocab_size = len(self.vocab_to_int)
			print("vocab_size", self.vocab_size)
		else:
			logger.error("No vocab_to_int data found, please re-run DataCenter.")
			exit()

	def __split_data(self, inputs, targets):
		'''
		Split data into Train, Val, Test according to 2:1:1
		:param inputs:
		:param targets:
		:return:
		'''
		cut_train = int(0.8 * len(inputs))
		cut_validation = int(0.9 * len(inputs))
		input_train, target_train = inputs[:cut_train], targets[:cut_train]
		input_val, target_val = inputs[cut_train:cut_validation], targets[cut_train:cut_validation]
		input_test, target_test = inputs[cut_validation:], targets[cut_validation:]
		return input_train, target_train, input_val, target_val, input_test, target_test

	def get_batches(self, x, y):
		number_batch = len(x) // self.batch_size
		for id_batch in range(0, number_batch, self.batch_size):
			start = id_batch * self.batch_size
			end = start + self.batch_size
			yield x[start:end], y[start:end]

	def build_inputs(self):
		"""
        Define placeholders for inputs, targets, and dropout
        :param batch_size: Batch size, number of sequences per batch
        :param num_steps: Number of sequence steps in a batch
        :return: tensorflow placeholders
        """
		# Declare placeholders we'll feed into the graph
		inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
		targets = tf.placeholder(tf.int32, [None, None], name='targets')

		# Keep probability placeholder for drop out layers
		keep_prob = tf.placeholder(tf.float32, name='keep_prob')

		return inputs, targets, keep_prob

	def build_encode_layer(self, input_data, keep_prob):

		# initialize weights based on random_uniform between [-1/sqrt(n), 1/sqrt(n)]
		# n is size of input to a neuron, rather than the batch size
		# limit = 1.0 / np.sqrt(self.vocab_size)
		limit = 0.01
		# enc_embed_input = tf.contrib.layers.embed_sequence(input_data, self.vocab_size, self.embed_size)
		print(self.vocab_size, self.embed_size)
		embedding = tf.Variable(tf.random_uniform((self.vocab_size, self.embed_size), -0.01, 0.01))
		enc_embed_input = tf.nn.embedding_lookup(embedding, input_data)

		def build_cell(lstm_size, keep_prob):
			# Use a basic LSTM cell
			lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

			# Add dropout to the cell
			drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
			return drop

		# Stack up multiple LSTM layers, consider multi layer of lstm as one cell
		enc_cell = tf.contrib.rnn.MultiRNNCell([build_cell(self.lstm_size, keep_prob) for _ in range(self.lstm_layer)])
		initial_state = enc_cell.zero_state(self.batch_size, tf.float32)
		enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, initial_state=initial_state,
		                                          dtype=tf.float32)
		return enc_output, enc_state

	def build_output(self, lstm_output):
		predictions = tf.contrib.layers.fully_connected(lstm_output[:, -1], 1, activation_fn=tf.sigmoid)
		predict = tf.identity(predictions, name="prediction")
		return predict

	def build_loss(self, predict, labels):
		cost = tf.losses.mean_squared_error(predictions=predict, labels=labels)
		cost = tf.reduce_mean(cost, name='cost')
		return cost

	def build_optimizer(self, cost):
		optimizer = tf.train.AdamOptimizer(self.learning_rate, name='optimizer').minimize(cost)
		return optimizer

	def build_graph(self):
		need_build_graph = False
		logger.info("Checking model graph...")
		if os.path.exists("{}.meta".format("{}/{}".format(self.save_model_path, self.model_name))):
			logger.info("Graph existed, ready to be reloaded...")
		else:
			need_build_graph = True
			logger.info("Graph not existed, create a new graph and save to {}".format(self.save_model_path))
			tf.reset_default_graph()

			inputs, targets, keep_prob = self.build_inputs()
			enc_output, _ = self.build_encode_layer(input_data=inputs, keep_prob=keep_prob)

			predict = self.build_output(enc_output)

			cost = self.build_loss(predict=predict, labels=targets)
			optimizer = self.build_optimizer(cost=cost)
			correct_prediction = tf.equal(tf.cast(tf.round(predict), tf.int32), targets)
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

			saver = tf.train.Saver()
			if not os.path.exists(self.save_model_path):
				os.mkdir(self.save_model_path)
			with tf.Session(graph=tf.get_default_graph()) as sess:
				sess.run(tf.global_variables_initializer())
				saver.save(sess, "{}/{}".format(self.save_model_path, self.model_name))

		logger.info("Finish building model graph!")
		return need_build_graph

	def load_graph(self, sess, need_build_graph):

		graph = defaultdict()

		# if the model graph just built from __build_graph, then no need to import meta graph again,
		# else, import the pre-built model graph.
		if need_build_graph:
			loader = tf.train.Saver()
		else:
			loader = tf.train.import_meta_graph("{}/{}".format(self.save_model_path, self.model_name) + '.meta')

		graph['inputs'] = sess.graph.get_tensor_by_name("inputs:0")
		graph['targets'] = sess.graph.get_tensor_by_name("targets:0")
		graph['keep_prob'] = sess.graph.get_tensor_by_name("keep_prob:0")
		graph['cost'] = sess.graph.get_tensor_by_name('cost:0')
		graph['prediction'] = sess.graph.get_tensor_by_name("prediction:0")
		graph['optimizer'] = sess.graph.get_operation_by_name("optimizer")
		graph['accuracy'] = sess.graph.get_tensor_by_name("accuracy:0")

		logger.info("model is ready, good to go!")

		check_point = tf.train.latest_checkpoint('checkpoints')
		# if no check_point found, means we need to start training from scratch, just initialize the variables.
		if not check_point:
			# Initializing the variables
			logger.info("Initializing the variables")
			sess.run(tf.global_variables_initializer())
		else:
			logger.info("check point path:{}".format(check_point))
			loader.restore(sess, check_point)
		return graph

	def get_batch(self, x, y):

		num_batch = len(x) // self.batch_size
		for start in range(0, len(x), self.batch_size):
			yield x[start:start + self.batch_size], y[start:start + self.batch_size]

	def validate_model(self, sess, inputs, targets, keep_prob, accuracy, x, y):
		res = []
		for val_x, val_y in self.get_batch(x, y):
			accuracy_cur = sess.run(accuracy, feed_dict={
				inputs: val_x,
				targets: val_y,
				keep_prob: 1.0
			})
			res.append(accuracy_cur)
		if res:
			logger.info("Validation/Test accuracy is:{}".format(np.mean(res)))

	@TaskReporter("Train graph")
	def train(self, inputs, targets):

		input_train, target_train, input_val, target_val, input_test, target_test = self.__split_data(inputs, targets)
		need_build_graph = self.build_graph()

		with tf.Session(graph=tf.get_default_graph()) as sess:
			graph = self.load_graph(sess, need_build_graph)
			saver = tf.train.Saver()
			logger.info("Start training...")
			iteration = 1
			for epoch in range(self.epochs):
				saver = tf.train.Saver()
				for x, y in self.get_batch(input_train, target_train):
					print(x, y)
					feed = {
						graph['inputs']: x,
						graph['targets']: y,
						graph['keep_prob']: self.keep_prob
					}
					cost, _, prediction = sess.run([graph['cost'], graph['optimizer'],
					                                graph['prediction']], feed_dict=feed)
					if iteration % 1 == 0:
						print("DSDFSDF")
						logger.info("Epoch: {}/{}\t".format(epoch + 1, self.epochs) +
						            "Iteration: {}\t".format(iteration) +
						            "Train loss: {:.3f}\t".format(cost))

					if iteration % self.validation_time == 0:
						self.validate_model(sess, graph['inputs'], graph['targets'],
						                    graph['keep_prob'], graph['accuracy'],
						                    input_val, target_val)
					if iteration % self.checkpoint_time == 0:
						saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(iteration, self.lstm_size))
					iteration += 1

			# Running the test data to see results
			self.validate_model(sess, graph['inputs'], graph['targets'], graph['keep_prob'], graph['accuracy'],
			                    input_val, target_val)

			saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(iteration, self.lstm_size))

	@TaskReporter("Test graph")
	def inference(self, input_data):
		print(input_data.shape)
		need_build_graph = self.build_graph()
		with tf.Session(graph=tf.get_default_graph()) as sess:
			graph = self.load_graph(sess, need_build_graph)
			logger.info("Start predicting...")
			prediction = sess.run(tf.round(graph['prediction']),
			                      feed_dict={
				                      graph['inputs']: input_data,
				                      graph['keep_prob']: 1.0
			                      }
			                      )
			logger.info("Predicted result:{}".format(prediction))
			return prediction
