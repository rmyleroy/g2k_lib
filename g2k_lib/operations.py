# -*- coding: utf-8 -*-

from __future__ import print_function, division
from transforms import filtering, dct2d, ks, ksinv, starlet2d
from scipy.ndimage import binary_erosion
from objects import Image, ComputedKappa, ResultRegister
from metrics import get_error
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


def generate_constraint(mask, bpix, dilation=False, i=None, Niter=None):
    """
        Generate a constraint matrix to be applied over the B mode;
        it is build using some erosion methods on the given mask.

        Parameters
        ----------
            mask : array_like
                Mask relative to missing data.
            bpix : int
                Defines the size of the structuring element.

        Todo
        ----
            Add documentation regarding the dilation

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

    else:
        if dilation:
            bpix = int(int(bpix) * (1 - (i - 1) / float(Niter - 1)))
        else:
            bpix = int(bpix)
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
    print((np.linalg.norm(next_kE - kE), np.linalg.norm(next_kB - kB)))
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
    print("Initialization.") if verbose else None
    kE, kB = np.zeros_like(g1map), np.zeros_like(
        g2map)  # A first estimate of kappa maps used as initialization
    if not dilation:
        print("Constraint construction.") if verbose else None
        constraint = generate_constraint(mask, bpix)
    range_ = tqdm(range(1, Niter + 1)) if tqdm_import else range(1, Niter + 1)
    for i in range_:
        print("Next step evaluation from iteration {}.".format(i - 1))
        kE, kB = next_step_gradient(kE, kB, g1map, g2map, mask, reduced)

        if dilation:
            constraint = generate_constraint(
                mask=mask, bpix=bpix, dilation=dilation, i=i, Niter=Niter + 1)

        if dct:
            if i == 1:
                # The maximum value of the DCT transform of the E-mode kappa map
                # is used as the maximum threshold value
                max_threshold = np.max(dct2d(kE))
                min_threshold = 0
            pm.dct_inpaint(kE=kE, i=i, Niter=Niter,
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
    print("fetching shear maps from {}".format(gamma_path)) if verbose else None
    gammas = Image.from_fits(gamma_path)  # Loads gamma from fits file

    g1map = gammas.get_layer(
        0) if no_padding else add_padding(gammas.get_layer(0))
    g2map = gammas.get_layer(
        1) if no_padding else add_padding(gammas.get_layer(1))
    if g1map.shape != g2map.shape:
        raise ValueError("Different shear map shapes: Got {} and {}.".format(
            g1map.shape, g2map.shape))
    if mask_path:
        print("fetching mask map from {}".format(
            mask_path)) if verbose else None
        # Loads mask from fits file
        mask = Image.from_fits(mask_path).get_layer() if no_padding else add_padding(
            Image.from_fits(mask_path).get_layer())
    else:
        print("No mask to be applied.") if verbose else None
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


def compute_errors(computed_kappa_path, gnd_truth_path, output_path=None):
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
    computed_kappa = ComputedKappa.from_fits(computed_kappa_path)
    # Loads data from the ground truth kappa file
    gnd_truth = Image.from_fits(gnd_truth_path)
    # Loads the mask used to compute the estimated kappa map.
    mask = Image.from_fits(computed_kappa.mask_path)
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
        if norm(gndB):
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

    computed_kappa.header['ERROR_E'] = get_error(
        diff[mask.get_layer().astype(bool)], gnd_truth.get_layer()[mask.get_layer().astype(bool)])
    computed_kappa.header['ERROR_B'] = get_error(
        diffB[mask.get_layer().astype(bool)], denomB[mask.get_layer().astype(bool)])

    computed_kappa.save()

    if not output_path:
        output_path = gnd_truth_path.replace(
            'inputs', 'outputs').replace('.fits', '.json')

    rr = ResultRegister(output_path)
    rr.set_error(computed_kappa.method, computed_kappa.niter, computed_kappa.bconstraint,
                 computed_kappa.header['ERROR_E'], 'e')
    rr.set_error(computed_kappa.method, computed_kappa.niter, computed_kappa.bconstraint,
                 computed_kappa.header['ERROR_B'], 'b')
    rr.save()

    return Image(np.array([diff, diffB]))
