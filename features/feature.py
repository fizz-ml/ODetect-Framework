class Feature(object):
    def __init__(self,sampling_rate,in_features,parameter_dict):
        """Initialize the feature with parameters specified by
        the parameter dictionary.
        Args:
            sampling_rate: the rate at which data is sampled from the thermistor

            parameter_dict: paramters specific to the features in the
            form of a dictionary
        """
        self._in_features = in_features
        # Syntactic sugar: if there are no features given, replace with identity.
        if len(self._in_features) == 0:
            self._in_features = [IdentityWindowFeature()]
        self._sampling_rate = sampling_rate
        self.set_params(parameter_dict)

    def get_params(self):
        """Returns a dictionary with the current params"""
        param_temp = self.get_param_template()
        for key in param_temp.keys():
            self._params[key] = getattr(self, '_' + key)

        return self._params

    def set_params(self, params):
        """ Set attributes using the supplied dictionary"""
        # Reset internal params
        self._params = {}

        # Get the paramater template to check against
        param_temp = self.get_param_template()

        # Go over the keys to check each param
        for key in param_temp.keys():
            param_temp_tuple = param_temp[key]
            param_type = param_temp_tuple[0]

            val = params.get(key)
            if val is None:
                # Check if the template defines a default value
                if len(param_temp_tuple) >= 3:
                    val = param_temp_tuple[2]
                else:
                    raise KeyError("Param {} was not specified.".format(key))

            # Cast param to the correct type if one was specified
            if param_type is not None:
                param_type = param_temp[key][0]
                val = param_type(val)

            # Add it to the param dictionary
            self._params[key] = val

            # Also add it as private fields to the algorithm object
            setattr(self, '_' + key, val)

    def get_param_template(self):
        """Returns a dicitionary template with a description for each param

        The keys of the dictionary template are the names of the params and
        the values are a pair of the expected type of the param and a
        brief description of the param."""
        raise NotImplementedError

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

class IdentityWindowFeature(WindowFeature):
    def __init__(self,sampling_rate=0,in_features=[],parameter_dict=[]):
        pass

    def get_param_template(self):
        """Returns a dicitionary template with a description for each param

        The keys of the dictionary template are the names of the params and
        the values are a pair of the expected type of the param and a
        brief description of the param."""
        return {}

    def calc_feature(self, window):
        return window
