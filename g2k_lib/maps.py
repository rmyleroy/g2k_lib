# -*- coding: utf-8 -*-

from objects import Image
from PIL import Image as Im
import matplotlib.mlab as mlab
import numpy as np


def normal_gauss(size, x_pos, y_pos, std_x=1.0, std_y=1.0):
    """
        Bivariate Gaussian distribution map.

        Parameters
        ----------
            size : int
                Size of the image in pixels.
            x_pos : float
                Mean value along the x coordinate.
            y_pos : float
                Mean value along the y coordinate.
            std_x : float, optional
                Standard deviation along the x coordinate.
            std_y : float, optional
                Standard deviation along the y coordinate.

        Returns
        -------
            2d-array
                Binned bivariate Gaussian distribution.

    """
    delta = 1  # granularity of the image
    x = y = np.arange(0, size, delta)
    X, Y = np.meshgrid(x, y)
    return mlab.bivariate_normal(X, Y, sigmax=std_x, sigmay=std_y, mux=x_pos, muy=y_pos)


def white_noise_map(size, std=1.0):
    """
        Noise map with normal distribution.

        Parameters
        ----------
            size : int
                Size of the image in pixels.
            std : float, optional
                Standard deviation of the distribution.

        Returns
        -------
            2d-array
                White noise map.

    """
    return np.random.randn(size * size).reshape((size, size)) * std


def resize_image(image, shape):
    """
        Returns a reshaped image.

        Parameters
        ----------
            image : Image
                Image to be resized.
            shape : tuple
                Shape of the new image.

        Returns
        -------
            Image
                A new image whose data are at the given shape.

        Note
        ----
            The values of the new data are determined using a bicubic interpolation algorithm.

    """
    resized = []
    for i in range(image.layers):
        im = Im.fromarray(image.get_layer(i))
        resized.append(np.array(im.resize(shape, resample=Im.BICUBIC)))
        print(resized)
    return Image(np.array(resized))
