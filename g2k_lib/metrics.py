# -*- coding: utf-8 -*-

import numpy as np


def get_error(diff, mask, truth, type_="std"):
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
    if type_ == "std":
        num = np.std(diff[mask.astype(bool)])
        denom = np.std(truth[mask.astype(bool)])
    if type_ == "norm":
        num = np.linalg.norm(diff[mask.astype(bool)])
        denom = np.linalg.norm(truth[mask.astype(bool)])
    return 100 * num / float(denum)
