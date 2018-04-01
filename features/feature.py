class Feature(object):
    def __init__(self,sampling_rate,parameter_dict):
        """Initialize the feature with parameters specified by 
        the parameter dictionary.
        Args:
            sampling_rate: the rate at which data is sampled from the thermistor

            parameter_dict: paramters specific to the features in the
            form of a dictionary
        """
        pass

    def save(path):
        """Save the parameters of the feature"""
        pass

    def load(path):
        """Load the parameters of the feature"""
        pass

class WindowFeature(Feature):
    def calc_feature(self, window):
        """
        calculates the feature across the entire input window
        returns an equally sized array
        
        Args:
            window: A 1 dimensional numpy array of float32s,
            representing the 
        """
        pass


class TrainableFeature(Feature):
    def train(training_data,validation_data):
        """
        trains the network on the trainining
        data and validation data. The training procedure is
        left up to the feature.
        
        Args:
            Training data: A list of 1 dimensional numpy arrays of float32s,
            representing the segments on which to train.

            Validation data: A list of 1 dimensional numpy arrays of float32s,
            that the feature can use as validation while training.
        """
        pass


class RTFeature(Feature):
    def push_x(self, x):
        """
        Pushes a new data point.
        
        Args: 
            x: a float32 representing the next datapoint
        """
        pass

    def get_feature(self):
        """
        Receive current value of the feature.
        """
        pass

