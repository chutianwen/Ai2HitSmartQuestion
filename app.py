import configparser

from DataCenter import DataCenter
from NeuralNetworks import NeuralNetworks


class App:
	def __init__(self):
		self.parser = configparser.ConfigParser()
		self.parser.read("config.INI")
		self.dataCenter = DataCenter(self.parser)
		self.neuralNetworks = NeuralNetworks(self.parser)

	def train(self):
		self.neuralNetworks.train(*self.dataCenter.process_data())

	def predict(self, data):
		"""
		API to predict the label of incoming question from user input
		:param data:
		:return:
		"""
		res = self.neuralNetworks.inference(self.dataCenter.process_inference_data(data))
		print(res)


if __name__ == "__main__":
	app = App()
	app.train()
	app.predict(["How is weather"])
