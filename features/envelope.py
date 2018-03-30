from features.feature import WindowFeature
from features.peakdetect import peakdetect
from scipy import interpolate
import numpy as np

class WindowEnvelopes(WindowFeature):
    """
    Estimates the evelope of the input feature and returns the amplitude.
    """
    def __init__(self):
        pass

    def calc_feature(self, window, lookahead=5, delta=0.02, s=0):
        """ Returns the max and min envelope of the window signal. """
        # Only grab the troughs not the peaks
        peak_idx_value, trough_idx_value = peakdetect(window, lookahead=lookahead, delta=delta)
        # Just need the position
        trough_idx = np.asarray([x[0] for x in trough_idx_value])
        trough_val = np.asarray([x[1] for x in trough_idx_value])

        peak_idx = np.asarray([x[0] for x in peak_idx_value])
        peak_val = np.asarray([x[1] for x in peak_idx_value])

        peak_envelope = self._envelope(peak_idx, peak_val, window.size, s=s)
        trough_envelope = self._envelope(trough_idx, trough_val, window.size, s=s)

        return [peak_envelope, trough_envelope]

    def _envelope(self, idx, val, window_size, s=0):
        return self._interpolate_spline(idx, val, np.arange(window_size), s=s)

    def _interpolate_spline(self, in_idx, in_val, out_idx, s=0):
        tck = interpolate.splrep(in_idx, in_val, s=s)
        return interpolate.splev(out_idx, tck, der=0)


class WindowEnvelopesAmplitude(WindowFeature):
    """
    Estimates the evelope of the input feature and returns the amplitude.
    """
    def __init__(self):
        self._envelope_feature = WindowEnvelopes()

    def calc_feature(self, window):
        """ Returns the max and min envelope of the window signal. """
        max_envelope, min_envelope = self._envelope_feature.calc_feature(window)
        return np.abs(max_envelope-min_envelope)
