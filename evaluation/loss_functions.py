import numpy as np

def CO2_timestamptofreq(X, sample_freq=300):
    """Takes an array of trough index-stamps and returns the frequency, computed using last n time-stamps
    """
    periods = np.empty_like(X)
    periods[1:] = np.diff(X) * 1/sample_freq

    # For the very first trough assume same period as next one
    periods[0] = periods[1]

    print(periods)

    return np.reciprocal(periods)


def RMSE(Y_, Y):
    """ Returns root mean squared error between Y_ and Y
    """
    return np.sqrt(np.square(Y_- Y).mean())

