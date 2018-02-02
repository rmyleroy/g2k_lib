# -*- coding: utf-8 -*-

from __future__ import print_function, division
from scipy.ndimage import binary_erosion
import numpy as np


def generate_constraint(mask, bpix):
    """
        Generate a constraint matrix to be applied over the B mode;
        it is build using some erosion methods on the given mask.

        Parameters
        ----------
            mask : array_like
                Mask relative to missing data.
            bpix : int
                Defines the size of the structuring element.

        Returns
        -------
            array_like
                The constraint matrix.

        Example
        --------
            >>> mask = numpy.array([[0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0]])
            >>> generate_constraint(mask, 1)
            array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])

    """
    if bpix == "None":
        return np.ones_like(mask)  # No constraint

    elif bpix == "Bzero":
        return np.zeros_like(mask)  # full constraint, 0 value everywhere

    elif str(bpix).isdigit:
        if int(bpix) < 0:
            raise ValueError("bpix cannot be negative.")
        else:
            bpix = int(bpix)
    else:
        raise ValueError("bpix must be 'None', 'Bzero' or an positive integer.")
    struct = np.ones((2 * bpix + 1, 2 * bpix + 1))  # Square structuring element
    # Erodes the mask to get the constraint matrix
    constraint = binary_erosion(mask, struct).astype(mask.dtype)

    return constraint


def cut_mask(data, mask):
    """
        Apply the mask over the data and cut data to get rid of the frame.

        Parameters
        ----------
            data : array_like
                Matrix with values to be extracted.
            mask : array_like
                Mask and data must have the same shape.

        Returns
        -------
            array_like
                Extracted version of the data matrix.

        Note
        -----
            The frame is determined by the outtermost nonzero values.

    """
    nz = np.nonzero(mask)
    xmax = np.max(nz[0])
    xmin = np.min(nz[0])
    ymax = np.max(nz[1])
    ymin = np.min(nz[1])
    return (data * mask)[xmin:xmax + 1, ymin:ymax + 1]


def add_padding(image):
    """
        Doubles the number of pixels along each axis by adding zero values arround the image.

        Parameters
        ----------
            image : array_like
                Image to be padded.

        Returns
        -------
            array_like
                Padded image.

        Examples
        --------

            # Odd number of pixels
            >>> image = numpy.array([[1]])
            >>> add_padding(image)
            array([[0., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 0.]])

            # Even number of pixels
            >>> image = numpy.array([[1,2],
                                     [3,4]])
            >>> add_padding(image)
            array([[0., 0., 0., 0.],
                   [0., 1., 2., 0.],
                   [0., 3., 4., 0.],
                   [0., 0., 0., 0.]])

        See Also
        --------
            remove_padding : Inverse function.
    """
    if len(image.shape) == 2:
        nx, ny = image.shape
        image_ = np.zeros((nx * 2 + nx % 2, ny * 2 + ny % 2))
        image_[int(nx / 2) + nx % 2:-int(nx / 2), int(ny / 2) + ny %
               2:-int(ny / 2)] = image
        return image_
    else:
        raise ValueError("Image must have at least 2 dimensions.")


def remove_padding(image):
    """
        Reduces by half the size of the image by removing border pixels.

        Parameters
        ----------
            image : array_like
                Image to be unpadded.

        Returns
        -------
            array_like
                Unpadded image.

        Examples
        --------

            >>> image = numpy.array([[0., 0., 0., 0.],
                                     [0., 1., 2., 0.],
                                     [0., 0., 0., 0.]])
            >>> remove_padding(image)
            array([[1., 2.])

        See Also
        --------
            add_padding : Inverse function.
    """
    if len(image.shape) == 2:
        nx, ny = image.shape
        image_ = image[int(nx / 4) + nx % 2:-int(nx / 4),
                       int(ny / 4) + ny % 2:-int(ny / 4)]
        return image_
    else:
        raise ValueError("Image must have at least 2 dimensions.")
