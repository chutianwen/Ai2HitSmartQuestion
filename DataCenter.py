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
		self.path_vocab_to_int = config_parser['Path']['path_vocab_to_int']
		self.path_int_to_vocab = config_parser['Path']['path_int_to_vocab']

	def process_data(self):

		with open(self.data_path) as f:
			text = f.read()

		# remove all the punctuations in the reviews
		# translator = str.maketrans('', '', string.punctuation)
		# text_rinsed = text.translate(translator)
		records = text.split("\n")

		if os.path.exists(self.path_vocab_to_int) and os.path.exists(self.path_int_to_vocab):
			logger.info("vocab_to_int already exist, just reload")
			vocab_to_int = np.load(self.path_vocab_to_int).item()
		else:
			# split by word
			clean_text = text.replace("\n", " ").replace("0", " ").replace("1", " ").replace(":", " ")
			vocabulary = set(clean_text.split())
			vocab_to_int = {vocab: idx for idx, vocab in enumerate(vocabulary, 1)}
			int_to_vocab = {idx: vocab for idx, vocab in enumerate(vocabulary, 1)}
			vocab_to_int['Padding'] = 0
			int_to_vocab[0] = "Padding"
			np.save(self.path_vocab_to_int, vocab_to_int)
			np.save(self.path_int_to_vocab, int_to_vocab)

		# print(vocab_to_int)
		# print(int_to_vocab)
		num_records = len(records)
		inputs = np.zeros([num_records, self.num_sequence], dtype=np.int32)
		targets = np.zeros([num_records, 1], dtype=np.int32)

		# If record is too long, chop the size based on num_sequence, else, padding 0 to the end.
		for idx in range(num_records):
			record_body, targets[idx][0] = records[idx].split(":")
			# print(record_body)
			record_words = record_body.split()
			num_words = len(record_words)
			non_zero_num = num_words if num_words <= self.num_sequence else self.num_sequence
			inputs[idx, :non_zero_num] = [vocab_to_int[word] for word in record_words[:non_zero_num]]

		# for x in zip(inputs, targets):
		# 	print(x)
		logger.info("Number of reviews:{}\tNumber of labels:{}".format(len(inputs), len(targets)))
		return inputs, targets

	def process_inference_data(self, questions):
		vocab_to_int = np.load(self.path_vocab_to_int).item()
		output_data = np.zeros([len(questions), self.num_sequence], dtype=np.int32)
		for id in range(len(questions)):
			words = questions[id].split()
			question_size = len(words)
			question_size = question_size if question_size <= self.num_sequence else self.num_sequence
			output_data[id, -question_size:] = [vocab_to_int[word] for word in words]
		# print(output_data)
		return output_data

	def run(self):
		return self.process_data()