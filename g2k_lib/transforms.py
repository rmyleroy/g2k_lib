# -*- coding: utf-8 -*-

from scipy.ndimage import convolve1d
from scipy.special import erf
from astropy.io import fits
from objects import Image
import matplotlib.pyplot as plt
import numpy as np
import os

IM_DCT_EXEC = os.path.abspath("./bin/im_dct")
_TMP_IN_DCT_PATH = os.path.abspath("/tmp/im_dct_in.fits")
_TMP_OUT_DCT_PATH = os.path.abspath("/tmp/im_dct_out.fits")


def dct2d(array, block_size=None, overlap=False):
    """
    Call the im_dct C++ routine to compute the corresponding DCT.
    """
    if os.path.exists(_TMP_IN_DCT_PATH):
        os.remove(_TMP_IN_DCT_PATH)
    if os.path.exists(_TMP_OUT_DCT_PATH):
        os.remove(_TMP_OUT_DCT_PATH)
    Image(array).save(_TMP_IN_DCT_PATH)
    opt = ''
    block_size_max = array.shape[0]
    if not block_size:
        block_size = block_size_max
    if block_size:
        if type(block_size) == int and block_size <= block_size_max:
            opt += " -b {} ".format(block_size)
    if overlap:
        opt += " -O "
    exec_command = str.join(
        ' ', [IM_DCT_EXEC, opt, _TMP_IN_DCT_PATH, _TMP_OUT_DCT_PATH])
    err = os.system(exec_command)
    if err:
        raise EnvironmentError("im_dct missed the call.")
    return Image.from_fits(_TMP_OUT_DCT_PATH).get_layer()


def idct2d(array, block_size=None, overlap=False):
    """
    Returns the 2-dimensional inverse DCT applied to the given array.
    """
    if os.path.exists(_TMP_IN_DCT_PATH):
        os.remove(_TMP_IN_DCT_PATH)
    if os.path.exists(_TMP_OUT_DCT_PATH):
        os.remove(_TMP_OUT_DCT_PATH)
    Image(array).save(_TMP_IN_DCT_PATH)
    opt = ' -r'
    block_size_max = array.shape[0]
    if not block_size:
        block_size = block_size_max
    if block_size:
        if type(block_size) == int and block_size <= block_size_max:
            opt += " -b {} ".format(block_size)
    if overlap:
        opt += " -O "

    exec_command = str.join(
        ' ', [IM_DCT_EXEC, opt, _TMP_IN_DCT_PATH, _TMP_OUT_DCT_PATH])
    err = os.system(exec_command)
    if err:
        raise EnvironmentError("im_dct missed the call.")
    return Image.from_fits(_TMP_OUT_DCT_PATH).get_layer()


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


def filtering(signal, threshold, block_size=None, overlap=False, norm=0):
    """
    This method return the input signal after its DCT components have been filtered.
    """

    block_size_max = signal.shape[0]
    if not block_size:
        block_size = block_size_max
    alpha = dct2d(signal, block_size, overlap)
    th_alpha = alpha
    if norm == 0:
        hard_threshold(th_alpha, threshold)
    if norm == 1:
        soft_threshold(th_alpha, threshold)
    th_alpha[::block_size, ::block_size] = alpha[::block_size,
                                                 ::block_size]  # Keep the 0 frequency component
    return idct2d(th_alpha, block_size, overlap)


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
    return th_max - (th_max - th_min) * n / float(Niter)


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
    return th_min + (th_max - th_min) * (1 - erf(2.8 * n / Niter))


def ks(g1map, g2map):
    """
        Computes Kaiser-Squires inversion mass map from binned gamma maps.

        Parameters
        ----------
            g1map, g2map : array_like
                binned gamma maps.

        Return
        ------
            tuple of array_like
                E-mode and B-mode kappa maps.

        See Also
        --------
            ksinv : Invers of ks function.
    """
    # g1map and g2map should be the same size
    (nx, ny) = g1map.shape

    # Compute Fourier space grid
    # Note: need to reverse the order of nx, ny to achieve proper k1, k2 shapes
    k1, k2 = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))

    # Compute Fourier transforms of g1 and g2
    g1hat = np.fft.fft2(g1map)
    g2hat = np.fft.fft2(g2map)

    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    k2[0, 0] = 1  # avoid division by 0
    kEhat = (p1 * g1hat + p2 * g2hat) / k2
    kBhat = (p2 * g1hat - p1 * g2hat) / k2

    # Transform back to real space
    kEmap = np.fft.ifft2(kEhat).real
    kBmap = np.fft.ifft2(kBhat).real

    return kEmap, kBmap


def ksinv(kEmap, kBmap):
    """
        Compute inverse Kaiser-Squires from E-mode and B-mode kappa maps.

        Parameters
        ----------
            kEmap, kBmap : array_like
                binned kappa maps.

        Return
        ------
            tuple of array_like
                E-mode and B-mode gamma maps.

        See Also
        --------
            ks : Invers of ksinv function.
    """
    # kEmap and kBmap should be the same size
    (nx, ny) = kEmap.shape

    # Compute Fourier space grid
    # Note: need to reverse the order of nx, ny to achieve proper k1, k2 shapes
    k1, k2 = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))

    # Compute Fourier transforms of kE and kB
    kEhat = np.fft.fft2(kEmap)
    kBhat = np.fft.fft2(kBmap)

    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    k2[0, 0] = 1  # avoid division by 0
    g1hat = (p1 * kEhat + p2 * kBhat) / k2
    g2hat = (p2 * kEhat - p1 * kBhat) / k2

    # Transform back to real space
    g1map = np.fft.ifft2(g1hat).real
    g2map = np.fft.ifft2(g2hat).real

    return g1map, g2map


def k2g_fits(k_path, g_path, mask_path=None, noise_path=None):
    """
        Computes gamma maps from kappa maps in fits file.

        Parameters
        ----------
            k_path : str
                Path to the kappa map fits file.
            g_path : str
                Path to save the gamma map fits file.
            mask_path : str, optional
                Path to the mask map fits file.
            noise_path : str
                Path to the noise map fits file.

        Return
        ------
            None

        Notes
        -----
            * The noise is added to the gamma maps.
                * If the noise fits file contains only one noise map,the noise will be added to both gamma maps.
                * Else (2 noise maps) the first (resp. second) willbe added to the first (res. second) gamma map.
            * The mask is applied over each gamma map.

    """
    # Loads kappa maps from fits file.
    kappa = Image.from_fits(k_path)

    if mask_path:
        # Loads mask from fits file if given.
        mask = Image.from_fits(mask_path).get_layer()
    else:
        mask = 1

    if noise_path:
        # Loads noise map from fits file if given.
        noises = Image.from_fits(noise_path)
        if noises.layers == 2:
            n1, n2 = noises.get_layer(0), noises.get_layer(1)
        elif noises.layers == 1:
            n1 = n2 = noises.get_layer(0)
        else:
            print("Cannot handle noise with more than 2 layers.")
            return
    else:
        n1, n2 = 0, 0

    # Getting E-mode and B-mode kappa maps
    if kappa.layers == 1:
        k1map = kappa.get_layer()
        k2map = k1map * 0
    elif kappa.layers == 2:
        k1map, k2map = kappa.get_layer(0), kappa.get_layer(1)
    else:
        print("Cannot handle kappa with more than 2 layers.")
        return
    # Evaluates gamma maps
    g1map, g2map = ksinv(k1map, k2map)
    # Apply the mask and noise over the compute gamma maps...
    gamma = Image(np.array([(g1map + n1) * mask, (g2map + n2) * mask]))
    # ... Then it is stored in a fits file under the given name
    gamma.save(g_path)


def starlet2d(image, nscales=5):
    """Compute the multiscale 2D starlet transform of an image."""
    # Filter banks
    h = np.array([1, 4, 6, 4, 1]) / 16.
    # g = np.array([0, 0, 1, 0, 0]) - h

    # Setting for convolve1d in order to match output of mr_transform
    mode = 'nearest'

    # Initialize
    result = np.zeros((nscales + 1, image.shape[0], image.shape[1]))
    cj = image

    # Compute multiscale starlet transform
    for j in range(nscales):
        # Create j-level version of h for à trous algorithm
        if j > 0:
            hj = np.array([[x] + [0] * (2**j - 1) for x in h]).flatten()
            hj = hj[:-(2**j - 1)]
        else:
            hj = h
        # Smooth coefficients at scale j+1
        cjplus1 = convolve1d(convolve1d(cj, weights=hj, mode=mode, axis=0),
                             weights=hj, mode=mode, axis=1)
        # Wavelet coefficients at scale j
        result[j] = cj - cjplus1
        cj = cjplus1
    result[-1] = cj

    return result
