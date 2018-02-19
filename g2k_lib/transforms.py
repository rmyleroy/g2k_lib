# -*- coding: utf-8 -*-

from scipy.ndimage import convolve1d
from scipy.fftpack import dct, idct
from scipy.special import erf
from astropy.io import fits
from struct import add_padding, remove_padding
from objects import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os

IM_DCT_EXEC = os.path.abspath("./bin/im_dct")


def new_dct2d(image, blockshape=None, overlap=False, norm='isap', pad=False):
    """
    Compute the 2D Discrete Cosine Transform (type 2) of an image.

    Parameters
    ----------
    image : 2d-array
        Image to transform.
    norm : {None, 'ortho', 'isap'}, optional
        Normalization option for `scipy.fftpack.dct`. Default is 'isap'.
    blocksize : int, optional
        TODO
    overlap : bool, optional
        TODO

    NOTE: apparently norm=None has a problem. ??
    """
    if norm not in [None, 'ortho', 'isap']:
        print("Warning: invalid norm --> using isap.")
        norm = 'isap'

    # Determine output shape
    n, m = image.shape


    if blockshape is not None:
        if type(blockshape) is int:
            k = l = blockshape  # Length and height of the block
        elif type(blockshape) is tuple:
            if len(blockshape) == 1:
                k = l = blockshape[0]
            elif len(blockshape) == 2:
                k, l = blockshape
            else:
                raise ValueError("Length of blockshape parameter cannot exceed 3. Got {}.".format(len(blockshape)))
        else:
            raise TypeError("blockshape parameter must be a tuple, not '{}'.".format(type(blockshape)))
        if type(k) is not int or type(l) is not int:
            raise TypeError("blockshape items must be integers. Got {}.".format(blockshape))

    else:
        blocksize = (n, m)


    # Compute DCT on sub blocks
    if overlap:
        for ii in range(2 * n / blocksize - 1):
            for jj in range(2 * n / blocksize - 1):
                i1 = ii * blocksize
                j1 = jj * blocksize
                # print(i1, i2, j1, j2)
                # print(i1/2, i1/2 + blocksize, j1/2, j1/2 + blocksize)
                # print('------------------------')
                imsub = add_padding(image[i1 / 2:i1 / 2 + blocksize,
                                          j1 / 2:j1 / 2 + blocksize])

                i2, j2 = imsub.shape

                if norm in (None, 'ortho'):
                    result[ii*i2:(ii+1)*i2, jj*j2:(jj+1)*j2] = dct(dct(imsub, norm=norm, axis=0),
                                               norm=norm, axis=1)
                else:
                    result[ii*i2:(ii+1)*i2, jj*j2:(jj+1)*j2] = dct(dct(imsub, norm='ortho', axis=0),
                                               norm='ortho', axis=1)
                    result[ii*i2:(ii+1)*i2, jj*j2:(jj+1)*j2][:, 0] *= np.sqrt(2)
                    result[ii*i2:(ii+1)*i2, jj*j2:(jj+1)*j2][0, :] *= np.sqrt(2)
    else:
        for ii in range(n / blocksize):
            for jj in range(n / blocksize):
                i1 = ii
                i2 = ii + blocksize
                j1 = jj
                j2 = jj + blocksize
                # print(i1, i2, j1, j2)
                # print('------------------------')
                imsub = add_padding(image[i1:i2, j1:j2])
                i2, j2 = imsub.shape

                if norm in (None, 'ortho'):
                    result[ii*i2:(ii+1)*i2, jj*j2:(jj+1)*j2] = dct(dct(imsub, norm=norm, axis=0),
                                               norm=norm, axis=1)
                else:
                    result[ii*i2:(ii+1)*i2, jj*j2:(jj+1)*j2] = dct(dct(imsub, norm='ortho', axis=0),
                                               norm='ortho', axis=1)
                    result[ii*i2:(ii+1)*i2, jj*j2:(jj+1)*j2][:, 0] *= np.sqrt(2)
                    result[ii*i2:(ii+1)*i2, jj*j2:(jj+1)*j2][0, :] *= np.sqrt(2)

    return result


def dct2d(image, blocksize=None, overlap=False, norm='isap'):
    """
    Compute the 2D Discrete Cosine Transform (type 2) of an image.

    Parameters
    ----------
    image : 2d-array
        Image to transform.
    norm : {None, 'ortho', 'isap'}, optional
        Normalization option for `scipy.fftpack.dct`. Default is 'isap'.
    blocksize : int, optional
        TODO
    overlap : bool, optional
        TODO

    NOTE: apparently norm=None has a problem. ??
    """
    if norm not in [None, 'ortho', 'isap']:
        print("Warning: invalid norm --> using isap")
        norm = 'isap'

    # Determine output shape
    n = image.shape[0]
    if blocksize is not None:
        if blocksize == n:
            result = np.zeros_like(image)
        elif blocksize not in [n / 2, n / 4, n / 8]:
            print("Warning: invalid blocksize --> using {}".format(n))
            blocksize = n
            result = np.zeros_like(image)
        else:
            if overlap:
                size = 2 * n - blocksize
                result = np.zeros((size, size))
            else:
                result = np.zeros_like(image)
    else:
        blocksize = n
        result = np.zeros_like(image)

    # Compute DCT on sub blocks
    if overlap:
        for ii in range(2 * n / blocksize - 1):
            for jj in range(2 * n / blocksize - 1):
                i1 = ii * blocksize
                i2 = i1 + blocksize
                j1 = jj * blocksize
                j2 = j1 + blocksize
                # print(i1, i2, j1, j2)
                # print(i1/2, i1/2 + blocksize, j1/2, j1/2 + blocksize)
                # print('------------------------')
                imsub = image[i1 / 2:i1 / 2 + blocksize,
                              j1 / 2:j1 / 2 + blocksize]
                if norm in (None, 'ortho'):
                    result[i1:i2, j1:j2] = dct(dct(imsub, norm=norm, axis=0),
                                               norm=norm, axis=1)
                else:
                    result[i1:i2, j1:j2] = dct(dct(imsub, norm='ortho', axis=0),
                                               norm='ortho', axis=1)
                    result[i1:i2, j1:j2][:, 0] *= np.sqrt(2)
                    result[i1:i2, j1:j2][0, :] *= np.sqrt(2)
    else:
        for ii in range(0, n, blocksize):
            for jj in range(0, n, blocksize):
                i1 = ii
                i2 = ii + blocksize
                j1 = jj
                j2 = jj + blocksize
                # print(i1, i2, j1, j2)
                # print('------------------------')
                imsub = image[i1:i2, j1:j2]
                if norm in (None, 'ortho'):
                    result[i1:i2, j1:j2] = dct(dct(imsub, norm=norm, axis=0),
                                               norm=norm, axis=1)
                else:
                    result[i1:i2, j1:j2] = dct(dct(imsub, norm='ortho', axis=0),
                                               norm='ortho', axis=1)
                    result[i1:i2, j1:j2][:, 0] *= np.sqrt(2)
                    result[i1:i2, j1:j2][0, :] *= np.sqrt(2)

    return result


def idct2d(image, blocksize=None, overlap=False, norm="isap"):
    """
    Compute the inverse 2D Discrete Cosine Transform (type 2) of an image.

    Parameters
    ----------
    image : 2d-array
        Image to transform.
    norm : {None, 'ortho', 'isap'}, optional
        Normalization option for `scipy.fftpack.idct`. Default is 'isap'.
    blocksize : int, optional
        TODO
    overlap : bool, optional
        TODO

    NOTE: apparently norm=None has a problem.
    """
    if norm not in [None, 'ortho', 'isap']:
        print("Warning: invalid norm --> using isap")
        norm = 'isap'

    # Determine output shape
    n = image.shape[0]
    if blocksize is not None:
        if blocksize == n:
            result = np.zeros_like(image)
        else:
            if overlap:
                size = (n + blocksize) / 2
                result = np.zeros((size, size))
            else:
                result = np.zeros_like(image)
    else:
        blocksize = n
        result = np.zeros_like(image)

    # Compute inverse DCT on sub blocks
    if overlap:
        for ii in range(n / blocksize):
            for jj in range(n / blocksize):
                i1 = ii * blocksize
                i2 = i1 + blocksize
                j1 = jj * blocksize
                j2 = j1 + blocksize
                i1r = i1 / 2
                i2r = i1r + blocksize
                j1r = j1 / 2
                j2r = j1r + blocksize
                # print(i1, i2, j1, j2)
                # print(i1r, i2r, j1r, j2r)
                # print('------------------------')
                imsub = image[i1:i2, j1:j2]
                if norm == 'isap':
                    imsub[:, 0] /= np.sqrt(2)
                    imsub[0, :] /= np.sqrt(2)
                    result[i1r:i2r, j1r:j2r] += idct(idct(imsub, norm='ortho', axis=0),
                                                     norm='ortho', axis=1)

                else:
                    result[i1r:i2r, j1r:j2r] += idct(idct(imsub, norm=norm, axis=0),
                                                     norm=norm, axis=1)
        # Take averages
        step = blocksize / 2
        counts = np.ones_like(result)
        counts[step:-step, :step] = 2
        counts[step:-step, -step:] = 2
        counts[:step, step:-step] = 2
        counts[-step:, step:-step] = 2
        counts[step:-step, step:-step] = 4
        result /= counts
    else:
        for ii in range(0, n, blocksize):
            for jj in range(0, n, blocksize):
                i1 = ii
                i2 = ii + blocksize
                j1 = jj
                j2 = jj + blocksize
                imsub = image[i1:i2, j1:j2]
                if norm == 'isap':
                    imsub[:, 0] /= np.sqrt(2)
                    imsub[0, :] /= np.sqrt(2)
                    result[i1:i2, j1:j2] = idct(idct(imsub, norm='ortho', axis=0),
                                                norm='ortho', axis=1)
                else:
                    result[i1:i2, j1:j2] = idct(idct(imsub, norm=norm, axis=0),
                                                norm=norm, axis=1)

    return result


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
                                                 ::block_size]  # Keep the 0 frequency component for each block
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

        Note
        ----
            Written by A. Peel
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
    """
    Compute the multiscale 2D starlet transform of an image.

    Parameters
    ----------
    image : array_like
        Image to decompose.
    nscales : int
        Number of scales for the decomposition.

    Returns
    -------
    array_like of array_like
        Array of length equal to nscales, containing every scale of the decomposition.

    Note
    ----
        Sum up all scales together to get the original image back.
        Written by A. Peel.
    """
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
