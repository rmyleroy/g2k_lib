# -*- coding: utf-8 -*-

from __future__ import print_function, division
from transforms import filtering, dct2d, ks, ksinv, starlet2d
from metrics import get_error
from objects import Image
from struct import add_padding, remove_padding, generate_constraint
import projection_methods as pm
import numpy as np
import copy
import time

try:
    from tqdm import tqdm  # To display progression bars
except ImportError:
    print("module 'tqdm' not installed.")
    tqdm_import = False
else:
    tqdm_import = True


def next_step_gradient(kE, kB, g1map, g2map, mask, reduced):
    """
        Computes the gradient and evaluates the next kappa values.

        Parameters
        ----------
            kE : array_like
                Current kappa E-mode.
            kB : array_like
                Current kappa B-mode.
            g1map : array_like
                Gamma E-mode.
            g2map : array_like
                Gamma B-mode.
            mask : array_like
                Missing data mask.
            reduced : bool
                If set ``True``, ``g1map`` and ``g2map`` are considered as reduced shear maps,
                else they are considered as observed shear maps.

        Returns
        -------
            tuple of array_like
                Both mode (E and B) of the next kappa maps.

    """
    g1, g2 = ksinv(kE, kB)  # Used to compute the residual
    if reduced:
        # Residuals using reduced shear
        r1 = (g1map * (1 - kE) - g1) * mask
        r2 = (g2map * (1 - kB) - g2) * mask
    else:
        # Residuals using observed shear
        r1 = (g1map - g1) * mask
        r2 = (g2map - g2) * mask
    dkE, dkB = ks(r1, r2)
    next_kE, next_kB = kE + dkE, kB + dkB
    # print((np.linalg.norm(next_kE - kE), np.linalg.norm(next_kB - kB)))
    return next_kE, next_kB


def iterative(g1map, g2map, mask, Niter=1, bpix="None", relaxed=False, relax_type=pm.ERF, dct=False, dct_type=pm.ERF, block_size=None, overlap=False, sbound=False, reduced=False, dilation=False, verbose=False):
    """
        Iteratively computes next kappa maps according to the given method
        and the number of iterations.

        Parameters
        ----------
            g1map : array_like
                Gamma E-mode.
            g2map : array_like
                Gamma B-mode.
            mask : array_like
                Missing data mask.
            Niter : int, optional
                Total number of iteration to perform the reconstruction.
            bpix : int or str, optional
                Number of pixel that caracterise the erosion to generate the constraint matrix.
                No constraint if set to ``"None"`` (default).
                Constraint over the entire image if set to ``"Bzero"``.
            reduced : bool, optional
                If set ``True``, ``g1map`` and ``g2map`` are considered as reduced shear maps,
                else they are considered as observed shear maps (default).

        Returns
        -------
            tuple of array_like
                Both (E and B) computed kappa.

    """
    if verbose:
        print("Initialization.")
    kE, kB = np.zeros_like(g1map), np.zeros_like(
        g2map)  # A first estimate of kappa maps used as initialization
    if not dilation:
        if verbose:
            print("Constraint construction.")
        constraint = generate_constraint(mask, bpix)
    range_ = tqdm(range(1, Niter + 1)) if tqdm_import else range(1, Niter + 1)
    for i in range_:
        if verbose:
            print("Next step evaluation from iteration {}.".format(i - 1))
        kE, kB = next_step_gradient(kE, kB, g1map, g2map, mask, reduced)

        if dilation:
            bpix_dil = str(int(int(bpix) * (1 - (i - 1) / float(Niter))))
            constraint = generate_constraint(mask, bpix_dil)

        if dct:
            if i == 1:
                # The maximum value of the DCT transform of the E-mode kappa map
                # is used as the maximum threshold value
                max_threshold = np.max(dct2d(kE))
                min_threshold = 0
            kE = pm.dct_inpaint(kE=kE, i=i, Niter=Niter,
                                max_threshold=max_threshold, min_threshold=min_threshold, block_size=block_size, overlap=overlap, verbose=verbose)

        if sbound:
            kE = std_constraint(kE, mask)

        if relaxed:
            kB = pm.iks_relaxed(kB=kB, i=i, Niter=Niter,
                                constraint=constraint, mask=mask, relax_type=relax_type, verbose=verbose)

        else:
            kB = pm.iks(kB=kB, constraint=constraint)

    return kE, kB


def std_constraint(image, mask, nscales=5):
    # TODO Test of std_constraint
    scales = starlet2d(image=image, nscales=nscales)
    result = 0
    for scale in scales[:-1]:
        result += std_flattening(scale, mask)
    result += scales[-1]
    return result


def std_flattening(data, mask):
    # TODO Test of std_flattening
    if data.shape != mask.shape:
        raise ValueError("dimension ")
    std_out = data[mask.astype(bool)].std()
    std_in = data[~mask.astype(bool)].std()
    flat_data = data * \
        (mask + (~mask.astype(bool)).astype(int) * (std_out / std_in))
    return flat_data


def compute_kappa(gamma_path, mask_path, niter, bpix, relaxed, relax_type, dct, dct_type, dct_block_size, overlap, sbound, reduced, dilation, verbose, no_padding):
    """
        Returns the computed kappa corresponding to the given configuration.

        Parameters
        ----------
            config : Config
                Must contain the following attributes:
                    * gammas : str (path)
                    * mask : str (path)
                    * method : str
                    * niter : int
                    * bconstraint : int or str
                    * reduced : bool

        Returns
        -------
            Image
                Image with both modes of the computed kappa maps.

    """
    if verbose:
        print("Loading shear maps from {}".format(gamma_path))
    gammas = Image.from_fits(gamma_path)  # Loads gamma from fits file

    g1map = gammas.get_layer(
        0) if no_padding else add_padding(gammas.get_layer(0))
    g2map = gammas.get_layer(
        1) if no_padding else add_padding(gammas.get_layer(1))
    if g1map.shape != g2map.shape:
        raise ValueError("Different shear map shapes: Got {} and {}.".format(
            g1map.shape, g2map.shape))
    if mask_path:
        if verbose:
            print("Loading mask map from {}".format(
                mask_path))
        # Loads mask from fits file
        mask = Image.from_fits(mask_path).get_layer() if no_padding else add_padding(
            Image.from_fits(mask_path).get_layer())
    else:
        if verbose:
            print("No mask to be applied.")
        mask = np.ones_like(g1map)
    if mask.shape != g1map.shape:
        raise ValueError("Cannot proceed with mask and shear maps of different shape: Got {} and {}.".format(
            mask.shape, g2map.shape))

    # Estimates kappa maps
    kE, kB = iterative(g1map=g1map, g2map=g2map, mask=mask, Niter=niter, bpix=bpix, relaxed=relaxed, relax_type=relax_type, dct=dct,
                       dct_type=dct_type, block_size=dct_block_size, overlap=overlap, sbound=sbound, reduced=reduced, dilation=dilation, verbose=verbose)
    kE = kE if no_padding else remove_padding(kE)
    kB = kB if no_padding else remove_padding(kB)
    data = np.array([kE, kB])
    return Image(data)


def compute_errors(computed_kappa_path, mask_path, gnd_truth_path, error_type):
    """
        Evaluates both errors (E and B), then stores the values in the computed kappa header
        and in a global results register.

        Parameters
        ----------
            computed_kappa_path : str
                Path toward the computed kappa fits file.
            gnd_truth_path : str
                Path toward the ground truth map fits file.
            output_path : str, optional
                Path toward the file in which the results register will be stored.

        Return
        ------
            Image
                Image of the difference between the computed and the ground truth kappa maps
                for both modes.
    """
    # Loads data from the computed kappa file.
    computed_kappa = Image.from_fits(computed_kappa_path)
    # Loads data from the ground truth kappa file
    gnd_truth = Image.from_fits(gnd_truth_path)
    # Loads the mask used to compute the estimated kappa map.
    mask = Image.from_fits(mask_path)
    # Check if there is a B mode to consider in the error computation.
    layers_truth = gnd_truth.layers
    if layers_truth == 1:
        # If there is only the E-mode, then B-mode is zero...
        gndB = gnd_truth.get_layer() * 0
        # ... and we compute the B-mode error according to the E-mode.
        denomB = gnd_truth.get_layer()
    elif layers_truth == 2:
        # If there is a B-mode...
        gndB = gnd_truth.get_layer(1)
        if np.linalg.norm(gndB):
            # ... and non zero, then we compute the B-mode error according to it, ...
            denomB = gndB
        else:
            # else we compute the B-mode error according to the E-mode.
            denomB = gnd_truth.get_layer()
    else:
        print('Cannot handle ground truth with more than 2 layers')
        return

    # E-mode difference.
    diff = computed_kappa.get_layer(0) - gnd_truth.get_layer(0)
    diffB = computed_kappa.get_layer(1) - gndB  # B-mode difference.

    errorE = get_error(
        diff, mask.get_layer(), gnd_truth.get_layer(), error_type)
    errorB = get_error(
        diffB, mask.get_layer(), denomB, error_type)

    return errorE, errorB
