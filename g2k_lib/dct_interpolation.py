# -*- coding: utf-8 -*-

from scipy.fftpack import dct, idct
from scipy.special import erf
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


def dct2d(array, norm='ortho'):
    """
    The 2-dimensional DCT applied to the given array.
    """
    return dct(dct(array.T, norm=norm).T, norm=norm)


def idct2d(array, norm='ortho'):
    """
    Returns the 2-dimensional inverse DCT applied to the given array.
    """
    return idct(idct(array.T, norm=norm).T, norm=norm)


def hard_threshold(array, value):
    """
    Modifies the given array by applying the hard-thresholding function where value is the threshold value.
    """
    array[(-np.abs(value) < array) & (np.abs(value) > array)] = 0


def soft_threshold(array, value):
    """
    Modifies the given array by applying the soft-thresholding function where value is the threshold value.
    """
    hard_threshold(array, value)
    array -= np.sign(array) * value


def filtering(signal, threshold, norm=0):
    """
    This method return the input signal after its DCT components have been filtered.
    """
    alpha = dct2d(signal)
    th_alpha = alpha
    if norm == 0:
        hard_threshold(th_alpha, threshold)
    if norm == 1:
        soft_threshold(th_alpha, threshold)
    th_alpha[0, 0] = alpha[0, 0]  # Keep the 0 frequency component
    return idct2d(th_alpha)


def get_threshold_n_lin(n, Niter, th_max, th_min):
    """
        Linearly decreasing thresholding function.

        Parameters
        ----------
            n : int
                Current iteration number.
            Niter : int
                Total iteration number.
            th_max : float
                Maximum thresholding value.
            th_min : float
                Minimum thresholding value.

        Return
        ------
            float
                Thresholding value for the current iteration.

    """
    return th_max - (th_max - th_min) * n / (Niter - 1)


def get_threshold_n_erf(n, Niter, th_max, th_min):
    """
        Exponentially decreasing thresholding function.

        Parameters
        ----------
            n : int
                Current iteration number.
            Niter : int
                Total iteration number.
            th_max : float
                Maximum thresholding value.
            th_min : float
                Minimum thresholding value.

        Return
        ------
            float
                Thresholding value for the current iteration.

    """
    return max(th_min + (th_max - th_min) * (1 - erf(2.8 * n / Niter)), th_min)
