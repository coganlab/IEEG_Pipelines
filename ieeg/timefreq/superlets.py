# Time-frequency analysis with superlets
# Based on 'Time-frequency super-resolution with superlets'
# by Moca et al., 2021 Nature Communications
#
# Implementation by Harald Bârzan and Richard Eugen Ardelean

#
# Note: for runs on multiple batches of data, the class SuperletTransform can be instantiated just once
# this saves time and memory allocation for the wavelets and buffers
#


import numpy as np
from scipy.signal import fftconvolve
from ieeg import Signal
import mne
from joblib import Parallel, delayed

# spread, in units of standard deviation, of the Gaussian window of the
# Morlet wavelet
MORLET_SD_SPREAD = 6

# the length, in units of standard deviation, of the actual support window of
# the Morlet
MORLET_SD_FACTOR = 2.5


def computeWaveletSize(fc, nc, fs):
    """
    Compute the size in samples of a morlet wavelet.

    Parameters
    ----------
    fc : float
        Center frequency in Hz.
    nc : float
        Number of cycles.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    int
        Size of the wavelet in samples.
    """
    sd = (nc / 2) * (1 / np.abs(fc)) / MORLET_SD_FACTOR
    return int(2 * np.floor(np.round(sd * fs * MORLET_SD_SPREAD) / 2) + 1)


def computeLongestWaveletSize(fs, foi, c1, ord):
    """
    Estimates the size of the longest wavelet.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    foi : array_like
        Frequencies of interest in Hz.
    c1 : float
        Base number of cycles parameter.
    ord : tuple or list
        The order or order range for superlets.

    Returns
    -------
    int
        Size of the longest wavelet in samples.
    """
    # make order parameter
    if len(ord) == 1:
        ord = (ord, ord)
    # orders = np.linspace(start=ord[0], stop=ord[1], num=len(foi))
    orders = np.interp(foi, [min(foi), max(foi)], ord)
    # create wavelets
    max = 0
    for iFreq in range(len(foi)):
        centerFreq = foi[iFreq]
        nWavelets = int(np.ceil(orders[iFreq]))

        for iWave in range(nWavelets):
            # create morlet wavelet
            wlen = computeWaveletSize(centerFreq, fs, (iWave + 1) * c1)
            if wlen > max:
                max = wlen

    return max


def gausswin(size, alpha):
    """
    Create a Gaussian window.

    Parameters
    ----------
    size : int
        Size of the window in samples.
    alpha : float
        Parameter controlling the width of the window.

    Returns
    -------
    ndarray
        Gaussian window of specified size.
    """
    halfSize = int(np.floor(size / 2))
    idiv = alpha / halfSize

    t = (np.arange(size, dtype=np.float64) - halfSize) * idiv
    window = np.exp(-(t * t) * 0.5)

    return window


def morlet(fc, nc, fs):
    """
    Create an analytic Morlet wavelet.

    Parameters
    ----------
    fc : float
        Center frequency in Hz.
    nc : float
        Number of cycles.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    ndarray
        Complex Morlet wavelet.
    """
    size = computeWaveletSize(fc, nc, fs)
    half = int(np.floor(size / 2))
    gauss = gausswin(size, MORLET_SD_SPREAD / 2)
    igsum = 1 / gauss.sum()
    ifs = 1 / fs

    t = (np.arange(size, dtype=np.float64) - half) * ifs
    wavelet = gauss * np.exp(2 * np.pi * fc * t * 1j) * igsum

    return wavelet


def fractional(x):
    """
    Get the fractional part of the scalar value x.

    Parameters
    ----------
    x : float
        Input scalar value.

    Returns
    -------
    float
        Fractional part of x.
    """
    return x - int(x)


class SuperletTransform:
    """
    Class used to compute the Superlet Transform of input data.

    This class implements the superlet transform algorithm for time-frequency
     analysis as described in Moca et al., 2021.
    """

    def __init__(self,
                 inputSize,
                 samplingRate,
                 baseCycles,
                 superletOrders,
                 frequencyRange=None,
                 frequencyBins=None,
                 frequencies=None):
        """
        Initialize the superlet transform.

        Parameters
        ----------
        inputSize : int
            Size of the input in samples.
        samplingRate : float
            The sampling rate of the input signal in Hz.
        baseCycles : float
            Number of cycles of the smallest wavelet (c1 in the paper).
        superletOrders : tuple
            A tuple containing the range of superlet orders, linearly
             distributed along frequencyRange.
        frequencyRange : tuple
            Tuple of ascending frequency points, in Hz.
        frequencyBins : int
            Number of frequency bins to sample in the interval frequencyRange.
        frequencies : array_like, optional
            Specific list of frequencies - can be provided instead of
             frequencyRange (it is ignored in this case).
        """
        # clear to reinit
        self.clear()

        # initialize containers
        if frequencies is not None:
            frequencyBins = len(frequencies)
            frequencyRange = [np.min(frequencies), np.max(frequencies)]
            self.frequencies = frequencies
        else:
            self.frequencies = np.linspace(start=frequencyRange[0],
                                           stop=frequencyRange[1],
                                           num=frequencyBins)

        self.inputSize = inputSize
        self.orders = np.interp(self.frequencies, frequencyRange,
                                superletOrders)
        # self.orders = np.linspace(start=superletOrders[0],
        #                           stop=superletOrders[1], num=frequencyBins)
        self.convBuffer = np.zeros(inputSize, dtype=np.complex128)
        self.poolBuffer = np.zeros(inputSize, dtype=np.float64)
        self.superlets = []

        # create wavelets
        for iFreq in range(frequencyBins):
            centerFreq = self.frequencies[iFreq]
            nWavelets = int(np.ceil(self.orders[iFreq]))

            self.superlets.append([])
            for iWave in range(nWavelets):
                # create morlet wavelet
                self.superlets[iFreq].append(
                    morlet(centerFreq, (iWave + 1) * baseCycles, samplingRate))

    def __del__(self):
        """
        Destructor.

        Cleans up resources when the object is deleted.
        """
        self.clear()

    def clear(self):
        """
        Clear the transform.

        Resets all internal variables to None, freeing memory.
        """
        # fields
        self.inputSize = None
        self.superlets = None
        self.poolBuffer = None
        self.convBuffer = None
        self.frequencies = None
        self.orders = None

    def longestWaveletSize(self):
        """
        Return the size of the longest wavelet.

        Returns
        -------
        int
            Size of the longest wavelet in samples.
        """
        max = 0
        for s in self.superlets:
            for w in s:
                if w.shape[0] > max:
                    max = w.shape[0]
        return max

    def validTimeRegion(self):
        """
        Compute the start and end of the valid spectrum region.

        Returns
        -------
        tuple
            A tuple containing:

            - start : int
                The start of the valid time region.
            - end : int
                The end of the valid time region.
        """
        pad = self.longestWaveletSize() // 2
        start = self.inputSize + pad
        end = self.inputSize - pad
        return start, end

    def transform(self, inputData):
        """
        Apply the transform to a buffer or list of buffers.

        Parameters
        ----------
        inputData : ndarray
            An NDarray of input data. Can be a single buffer or a list of
             buffers.

        Returns
        -------
        ndarray
            The transformed data as a time-frequency representation.

        Raises
        ------
        Exception
            If input data size doesn't match the defined input size for this
             transform.
        """

        # compute number of arrays to transform
        if len(inputData.shape) == 1:
            if inputData.shape[0] != self.inputSize:
                raise ValueError("Input data must meet the defined input size"
                                 " for this transform.")

            result = np.zeros((self.inputSize, len(self.frequencies)),
                              dtype=np.float64)
            self.transformOne(inputData, result)
            return result

        else:
            n = int(np.sum(inputData.shape[0:len(inputData.shape) - 1]))
            insize = int(inputData.shape[len(inputData.shape) - 1])

            if insize != self.inputSize:
                raise ValueError("Input data must meet the defined input size"
                                 " for this transform.")

            # reshape to data list
            datalist = np.reshape(inputData, (n, insize), 'C')
            result = np.zeros((len(self.frequencies), self.inputSize),
                              dtype=np.float64)

            for i in range(0, n):
                self.transformOne(datalist[i, :], result)

            return result / n

    def transformOne(self, inputData, accumulator):
        """
        Apply the superlet transform on a single data buffer.

        Parameters
        ----------
        inputData : ndarray
            A 1xInputSize array containing the signal to be transformed.
        accumulator : ndarray
            A spectrum to accumulate the resulting superlet transform.

        Notes
        -----
        This method modifies the accumulator array in-place.
        """
        accumulator.resize((len(self.frequencies), self.inputSize))

        for iFreq in range(len(self.frequencies)):

            # init pooling buffer
            self.poolBuffer.fill(1)

            if len(self.superlets[iFreq]) > 1:

                # superlet
                nWavelets = int(np.floor(self.orders[iFreq]))
                rfactor = 1.0 / nWavelets

                for iWave in range(nWavelets):
                    self.convBuffer = fftconvolve(inputData,
                                                  self.superlets[iFreq][iWave],
                                                  "same")
                    self.poolBuffer *= 2 * np.abs(self.convBuffer) ** 2

                if fractional(self.orders[iFreq]) != 0 and len(
                        self.superlets[iFreq]) == nWavelets + 1:
                    # apply the fractional wavelet
                    exponent = self.orders[iFreq] - nWavelets
                    rfactor = 1 / (nWavelets + exponent)

                    self.convBuffer = fftconvolve(inputData,
                                                  self.superlets[iFreq][
                                                      nWavelets], "same")
                    self.poolBuffer *= (2 * np.abs(
                        self.convBuffer) ** 2) ** exponent

                # perform geometric mean
                accumulator[iFreq, :] += self.poolBuffer ** rfactor

            else:
                # wavelet transform
                accumulator[iFreq, :] += (2 * np.abs(
                    fftconvolve(inputData, self.superlets[iFreq][0],
                                "same")) ** 2).astype(np.float64)


def cropSpectrum(spectrum, paddingSize):
    """
    Remove paddingSize samples at both ends of the spectrum.

    Parameters
    ----------
    spectrum : ndarray
        A 2D numpy array representing the time-frequency spectrum.
    paddingSize : int
        Number of samples to remove - equals to longestWaveletSize() / 2
        of the computing SuperletTransform object.

    Returns
    -------
    ndarray
        The spectrum with the padding removed.
    """
    return spectrum[:, paddingSize:(spectrum.shape[1] - paddingSize)]


# main superlet function
def superlets(data,
              fs,
              foi,
              c1,
              ord):
    """
    Perform fractional adaptive superlet transform (FASLT) on a list of trials.

    Parameters
    ----------
    data : ndarray
        A numpy array of data. The rightmost dimension of the data is the trial
         size. The result will be the average over all the spectra.
    fs : float
        The sampling rate in Hz.
    foi : array_like
        List of frequencies of interest.
    c1 : float
        Base number of cycles parameter.
    ord : tuple or list
        The order (for SLT) or order range (for FASLT), spanned across the
        frequencies of interest.

    Returns
    -------
    ndarray
        A matrix containing the average superlet spectrum.

    Notes
    -----
    This is the main function for computing the superlet transform.
    """
    # determine buffer size
    bufferSize = data.shape[-1]

    # make order parameter
    if len(ord) == 1:
        ord = (ord, ord)

    # build the superlet analyzer
    faslt = SuperletTransform(inputSize=bufferSize,
                              frequencyRange=None,
                              frequencyBins=None,
                              samplingRate=fs,
                              frequencies=foi,
                              baseCycles=c1,
                              superletOrders=ord)

    # apply transform
    result = faslt.transform(data)
    faslt.clear()

    return result


def superlet_tfr(inst: Signal,
                 foi: list[float],
                 c1: float,
                 ord: tuple[int, int] = (1, 1),
                 decim: int = 1,
                 n_jobs: int = 1) -> Signal:
    """
    Compute the superlet time-frequency representation of the input signal.

    Parameters
    ----------
    inst : Signal
        The input signal (e.g., Raw, Epochs, Evoked).
    foi : list[float]
        List of frequencies of interest.
    c1 : float
        Base number of cycles parameter.
    ord : tuple[int, int], optional
        The order (for SLT) or order range (for FASLT), spanned across the
        frequencies of interest. Default is (1, 1).
    decim : int, optional
        Decimation factor for the output. Default is 1.

    Returns
    -------
    Signal
        The time-frequency representation of the input signal.
    """
    # check if the input is a Raw or Epochs object
    times = inst.times[::decim]
    sfreq = inst.info['sfreq']
    if isinstance(inst, (mne.io.BaseRaw | mne.Epochs)):
        data = inst.get_data()

    # check if the input is an Evoked object
    elif isinstance(inst, mne.Evoked):
        data = inst.data[np.newaxis, :]

    else:
        raise ValueError("Input must be a Raw, Epochs or Evoked object.")

    # compute superlet transform
    # determine buffer size
    bufferSize = data.shape[-1]

    # make order parameter
    if len(ord) == 1:
        ord = (ord, ord)

    # build the superlet analyzer
    faslt = SuperletTransform(inputSize=bufferSize,
                              frequencies=foi,
                              samplingRate=sfreq,
                              baseCycles=c1,
                              superletOrders=ord)
    freqs = faslt.frequencies
    out = np.zeros(data.shape[:-1] + (len(freqs), len(times)),
                   dtype=data.dtype)

    def _apply_transform(idx):
        tout = faslt.transform(data[idx])[..., ::decim]
        return tout, idx

    # apply transform in parallel
    par = Parallel(n_jobs=n_jobs, return_as='generator_unordered', verbose=10)(
        delayed(_apply_transform)(i) for i in np.ndindex(data.shape[:-1]))
    for o, i in par:
        # apply transform
        out[i] = o
    faslt.clear()

    # create TFR object and return it
    tfr = mne.time_frequency.EpochsTFRArray(inst.info, out, times, freqs,
                                            events=inst.events,
                                            event_id=inst.event_id)

    return tfr
