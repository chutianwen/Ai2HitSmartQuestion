import configparser

from AppUtils import logger
from DataCenter import DataCenter
from NeuralNetworks import NeuralNetworks

if __name__ == "__main__":

    parser = configparser.ConfigParser()
    parser.read("config.INI")
    dataCenter = DataCenter(parser)
    inputs, targets = dataCenter.run()
    neuralNetworks = NeuralNetworks(parser)
    neuralNetworks.train(inputs, targets)

    # test_data = ["How is weather"]
    # test_data_processed = dataCenter.process_inference_data(test_data)
    # neuralNetworks.inference(test_data_processed)
    logger.info("Job Done!")