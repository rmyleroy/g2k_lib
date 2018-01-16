# -*- coding: utf-8 -*-

import numpy as np


def norm(x, L=2):
    """
        Returns the L-norm of a matrix.

        Parameters
        ----------
            x : array_like
                The matrix whose norm wants to be computed.
            L : float or numpy.inf
                The order of the norm:
                    * L=0 : returns the amount of non zero values in x.
                    * L=numpy.inf : returns the maximum value in x.
                    * else : standard L-th norm.

        Return
        ------
        float
            The computed norm.

    """
    if type(x) is not np.ndarray:
        x = np.array(x)
    if L == 0:
        return len(x[x != 0])  # amount of non zero values in x
    elif L == np.inf:
        return np.max(x)
    else:
        return np.power(np.sum(np.power(np.abs(x), L)), 1 / float(L))


def get_error(num, denom):
    """
        Returns a percentage ratio between the two given array_like parameters
        according to their standard deviation.

        Parameters
        ----------
            num : array_like
                The numerator.
            denom : array_like
                The denominator.

        Return
        ------
            float
                The percentage ratio.
    """
    num = num.std()
    denom = denom.std()
    return 100 * num / float(denom)
