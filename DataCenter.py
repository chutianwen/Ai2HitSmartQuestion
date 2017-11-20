from AppUtils import logger
import numpy as np
import string
import os


class DataCenter:
	def __init__(self, config_parser):
		"""
		:param data_path: path storing text data
		:param batch_size:
		:param feature_length:
		"""
		self.data_path = config_parser['Path']['data_path']
		self.num_sequence = int(config_parser['ModelParas']['num_sequence'])


	def process_data(self):

		with open("{}/{}".format(self.data_path, "DummyData.txt")) as f:
			text = f.read()
		# remove all the punctuations in the reviews
		# translator = str.maketrans('', '', string.punctuation)
		# text_rinsed = text.translate(translator)
		reviews_separated = text.split("\n")

		path_int_to_vocab = "{}/int_to_vocab.npy".format(self.data_path)
		path_vocab_to_int = "{}/vocab_to_int.npy".format(self.data_path)
		if os.path.exists(path_vocab_to_int) and os.path.exists(path_int_to_vocab):
			logger.info("vocab_to_int already exist, just reload")
			vocab_to_int = np.load(path_vocab_to_int).item()
		else:
			# split by word
			vocabulary = set(text.replace(":", " ").split())
			vocab_to_int = {vocab: id for id, vocab in enumerate(vocabulary, 1)}
			int_to_vocab = {id: vocab for id, vocab in enumerate(vocabulary, 1)}
			np.save(path_vocab_to_int, vocab_to_int)
			np.save(path_int_to_vocab, int_to_vocab)
		features = np.zeros([len(reviews_separated), self.num_sequence], dtype=np.int32)
		targets = np.zeros([len(reviews_separated), 1], dtype=np.int32)
		for id in range(features.shape[0]):
			review_words, targets[id][0] = reviews_separated[id].split(":")
			review_words = review_words.split()
			review_size = len(review_words)
			review_size = review_size if review_size <= self.num_sequence else self.num_sequence
			features[id, -review_size:] = [vocab_to_int[word] for word in review_words]

		# print(features)
		# print(targets)
		logger.info("Number of reviews:{}\tNumber of labels:{}".format(len(features), len(targets)))
		return features, targets

	def process_inference_data(self, questions):
		path_vocab_to_int = "{}/vocab_to_int.npy".format(self.data_path)
		vocab_to_int = np.load(path_vocab_to_int).item()
		output_data = np.zeros([len(questions), self.num_sequence], dtype=np.int32)
		for id in range(len(questions)):
			words = questions[id].split()
			question_size = len(words)
			question_size = question_size if question_size <= self.num_sequence else self.num_sequence
			output_data[id, -question_size:] = [vocab_to_int[word] for word in words]
		print(output_data)
		return output_data

	def run(self):
		return self.process_data()