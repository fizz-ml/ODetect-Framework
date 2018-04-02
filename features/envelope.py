from features.feature import WindowFeature
from features.peakdetect import peakdetect
from features.simple_filter import normalize
from scipy import interpolate
import numpy as np


# TODO: This is not a proper feature compatible with the framework
# Just a helper for WinEnvAmp below
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
    def calc_feature(self, window):
        """ Returns the max and min envelope of the window signal. """
        self._envelope_feature = WindowEnvelopes()
        lookahead = int(np.floor(float(self._lookahead_length*self._sampling_rate)))
        max_envelope, min_envelope = self._envelope_feature.calc_feature(window, lookahead=lookahead, delta=self._delta)
        amplitude = np.abs(max_envelope-min_envelope)
        if self._normalize:
            return normalize(amplitude)
        else:
            return amplitude

    def get_param_template(self):
        param_template = {
            "lookahead_length": (float, "Distance to look ahead from \
                a peak candidate to determine if it is the actual \
                peak in seconds."), # Default was 5/200
            "delta": (float, "This specifies a minimum difference between \
                a peak and the following points, before a peak may be \
                considered a peak."), # Default was 0.02
            "normalize": (bool, "Whether to normalize the feature before returning it.", True)
            }
        return param_template
