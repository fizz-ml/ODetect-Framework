import numpy as np

class Buffer():
    """ Keeps a fixed window buffer of input objects.
    """
    def __init__(self, buffer_size, dtype=float):
        """ Instantiates a buffer object.
        Args:
            buffer_size (int): Specifies the maximum size of the buffer.
            dtype (type, optional): Specifies the type of data to be stored.
                Defaults to float.
        """
        self.buffer_size = buffer_size
        self._buffer = np.zeros(buffer_size, dtype=dtype)

    def reset(self):
        """ Resets the buffer. """
        self._buffer = np.zeros(buffer_size)

    def put(self, x):
        """ Puts a new value into the buffer. """
        self._buffer = np.roll(self._buffer,-1)
        self._buffer[-1] = x

    def get(self):
        """ Retrieves a copy of the current buffer.

        Returns a copy of the current buffer. Elements are ordered temporally
        with the oldest element at the start of the array and the most recent
        at the end. If the current buffer has not yet been filled to capacity,
        the start of the array will be padded with zero values.

        Args:
            None
        Returns:
            ndarray: Copy of the current buffer as an array of buffer_size.
        """
        return np.copy(self._buffer)

