from features.feature import WindowFeature
from features.peakdetect import peakdetect
from features.simple_filter import normalize
import numpy as np
from scipy import interpolate

class OldPeakTroughPoints():
    def __init__(self):
        pass

    def calc_feature(self, window, delta=1.0, lookahead=100):
        '''
            Returns the index and value and period between peaks and troughs for the supplied co2 signal.
        '''
        peaks_idx_value, troughs_idx_value = peakdetect(window, lookahead=lookahead, delta=delta)

        troughs_idx = np.asarray([x[0] for x in troughs_idx_value])
        troughs_val = np.asarray([x[1] for x in troughs_idx_value])
        peaks_idx = np.asarray([x[0] for x in peaks_idx_value])
        peaks_val = np.asarray([x[1] for x in peaks_idx_value])

        trough_period = self._calc_period(troughs_idx)
        peak_period = self._calc_period(peaks_idx)

        return ((peaks_idx, peaks_val, peak_period),(troughs_idx, troughs_val, trough_period))

    def _calc_period(self, idx):
        troughs_period = np.empty_like(idx)

        # For the very last trough assume same period as next one
        troughs_period[:-1] = np.diff(idx)
        troughs_period[-1] = troughs_period[-2]
        return troughs_period


class WindowPeakTroughPoints(WindowFeature):
    def __init__(self,sampling_rate,in_features,parameter_dict):
        super(WindowPeakTroughPoints,self).__init__(sampling_rate,in_features,parameter_dict)
        assert len(self._in_features) == 1, "WindowPeakTroughPoints only accepts one input feature, but {} given.".format(len(self._in_features))

    def calc_feature(self, window):
        '''
            Returns the index and value and period between peaks and troughs for the supplied co2 signal.
        '''
        window = self._in_features[0].calc_feature(window)
        lookahead = int(np.floor(float(self._lookahead_length*self._sampling_rate)))
        peaks_idx_value, troughs_idx_value = peakdetect(window, lookahead=lookahead, delta=self._delta)

        troughs_idx = np.asarray([x[0] for x in troughs_idx_value])
        troughs_val = np.asarray([x[1] for x in troughs_idx_value])
        peaks_idx = np.asarray([x[0] for x in peaks_idx_value])
        peaks_val = np.asarray([x[1] for x in peaks_idx_value])

        return ((peaks_idx, peaks_val),(troughs_idx, troughs_val))

    def get_param_template(self):
        param_template = {
            "lookahead_length": (float, "Distance to look ahead from a peak candidate to determine if it is the actual peak in seconds."), # Default was 5/200
            "delta": (float, "This specifies a minimum difference between a peak and the following points, before a peak may be considered a peak.") # Default was 0.02
            }
        return param_template

class WindowPeakTroughPeriods(WindowFeature):
    def __init__(self,sampling_rate,in_features,parameter_dict):
        super(WindowPeakTroughPeriods,self).__init__(sampling_rate,in_features,parameter_dict)
        assert len(self._in_features) == 1, "WindowPeakTroughPeriods only accepts one input feature, but {} given.".format(len(self._in_features))

    def calc_feature(self, window, interpolate=True):
        '''
        Returns the period between peaks and period between troughs.
        If interpolate is True, returns an interpolated signal of this value
        with the same length of the input window.
        '''
        window = self._in_features[0].calc_feature(window)
        lookahead = int(np.floor(float(self._lookahead_length*self._sampling_rate)))
        peaks_idx_value, troughs_idx_value = peakdetect(window, lookahead=lookahead, delta=self._delta)

        troughs_idx = np.asarray([x[0] for x in troughs_idx_value])
        peaks_idx = np.asarray([x[0] for x in peaks_idx_value])

        troughs_period = self._calc_period(troughs_idx)
        peaks_period = self._calc_period(peaks_idx)

        if interpolate:
            interpolator = None
            if self._interp == 'spline':
                interpolator = self._interpolate_spline
            elif self._interp == 'line':
                interpolator = self._interpolate_line
            """
            Old experiment to see if interpolating them together would work
            Averaging them seems to work better
            if joint:
                idxs = np.concatenate((peaks_idx, troughs_idx))
                vals = np.concatenate((peaks_period, troughs_period))
                sort_idxs = np.argsort(idxs)
                idxs = idxs[sort_idxs]
                vals = vals[sort_idxs]
                return interpolator(idxs, vals, np.arange(window.size), s=s)
            else:
            """
            interp_troughs_period = interpolator(troughs_idx,
                    troughs_period, np.arange(window.size),
                    s=self._s)
            interp_peaks_period = interpolator(peaks_idx,
                    peaks_period, np.arange(window.size),
                    s=self._s)

            output = (interp_peaks_period if self._toggle_p_t else interp_troughs_period)
            if self._normalize:
                return normalize(output)
            else:
                return output
        else:
            return ((peaks_idx, peaks_period),
                    (troughs_idx, troughs_period))

    def get_param_template(self):
        param_template = {
            "toggle_p_t": (bool, "If true returns peak else trough."), # TODO this should not be needed (it's due to assumption that features output 1d arrays rn)
            "lookahead_length": (float, "Distance to look ahead from \
                a peak candidate to determine if it is the actual \
                peak in seconds."), # Default was 5/200
            "delta": (float, "This specifies a minimum difference between \
                a peak and the following points, before a peak may be \
                considered a peak."), # Default was 0.02
            "interp": (str, "The type of interpolation \
                to be used for the output. \
                One of: ['spline', 'line']."), # Default was 'spline'
            "s": (float, "Regularization parameter for smoothing, \
                if using the spline interpolation."), # Default was 0
            "normalize": (bool, "Whether to normalize the feature before returning it.", True)
            }
        return param_template

    def _calc_period(self, idx):
        troughs_period = np.empty_like(idx)

        # For the very last trough assume same period as next one
        troughs_period[:-1] = np.diff(idx)
        troughs_period[-1] = troughs_period[-2]
        return troughs_period

    def _interpolate_spline(self, in_idx, in_val, out_idx, s=0):
        tck = interpolate.splrep(in_idx, in_val, s=s)
        return interpolate.splev(out_idx, tck, der=0)

    def _interpolate_line(self, in_idx, in_val, out_idx, s=0):
        return np.interp(out_idx, in_idx, in_val)

