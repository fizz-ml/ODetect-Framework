class Algorithm(object):
    def __init__(self, params):
        raise NotImplementedError

    def evaluate(self):
        """Return the next predicted point"""
        raise NotImplementedError

    def update(self, x):
        """Updates the model with a new data point"""
        raise NotImplementedError

    def reset(self):
        """Reset the model"""
        raise NotImplementedError

    def get_params(self):
        """Returns a dictionary with the current params"""
        return self._params

    def set_params(self, params):
        """Set a dictionary with the current params"""
        # Reset internal params
        self._params = {}

        # Get the paramater template to check against
        param_temp = get_param_template()

        # Go over the keys to check each param
        for key in param_temp.keys():
            val = params.get(key)
            if val is None:
                raise KeyError("Param {} was not specified.".format(key))

            # Convert param to the correct type
            param_type = param_temp(key)[0]
            val = param_type(val)

            # Add it to the param dictionary
            self._params[key] = val

    def get_param_template(self):
        """Returns a dicitionary template with a description for each param

        The keys of the dictionary template are the names of the params and
        the values are a pair of the expected type of the param and a
        brief description of the param."""
        """
        raise NotImplementedError
