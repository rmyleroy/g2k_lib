# -*- coding: utf-8 -*-

from __future__ import print_function, division
from dct_interpolation import filtering, dct2d
from scipy.ndimage import binary_erosion
from objects import Image, ComputedKappa, ResultRegister
from metrics import get_error, norm
import projection_methods as pm
import numpy as np
import copy
import time

try:
    from tqdm import *  # To display progression bars
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
    if bpix == 'None':
        return np.ones_like(mask)  # No constraint

    if bpix == 'Bzero':
        return np.zeros_like(mask)  # full constraint, 0 value everywhere

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
    g1, g2 = ks93inv(kE, kB)  # Used to compute the residual
    if reduced:
        # Residuals using reduced shear
        r1 = (g1map * (1 - kE) - g1) * mask
        r2 = (g2map * (1 - kB) - g2) * mask
    else:
        # Residuals using observed shear
        r1 = (g1map - g1) * mask
        r2 = (g2map - g2) * mask
    dkE, dkB = ks93(r1, r2)

    return kE + dkE, kB + dkB


def projection(method, next_kE, next_kB, i, Niter, max_threshold, min_threshold, constraint, mask):
    """
        Returns the correct projection of kappa maps according to the given method.

        Parameters
        ----------
            method : str
                Name of the method used to compute kappa maps.
            next_kE : array_like
                Kappa E-mode map to be projected.
            next_kB : array_like
                Kappa B-mode map to be projected.
            i : int
                Current iteration number.
            Niter : int
                Total number of iteration.
            max_threshold : float
                Maximum threshold value to be used for the DCT.
            min_threshold : float
                Minimum threshold value to be used for the DCT.
            constraint : array_like
                Constraint matrix.
            mask : array_like
                Missing data mask.

        Returns
        -------
            tuple of array_like
                Projected kappa maps (both E and B mode).

        Raises
        ------
            ValueError
                Raised in case of unexpected method value.

    """
    if method == pm.MethodNames.IKS:
        return pm.iks(next_kE, next_kB, constraint)

    elif method == pm.MethodNames.IKS_RELAXED:
        return pm.iks_relaxed(next_kE, next_kB, i, Niter, constraint, mask)

    elif method == pm.MethodNames.DCT:
        return pm.dct_inpaint(next_kE, next_kB, i, Niter, max_threshold, min_threshold, constraint, mask)

    elif method == pm.MethodNames.FULL_DCT:
        return pm.full_dct(next_kE, next_kB, i, Niter, max_threshold, min_threshold, constraint, mask)

    else:
        raise ValueError("Unexpected method value: {}".format(method))


def iterative(method, g1map, g2map, mask, Niter=20, bpix="None", reduced=False, dilation=False):
    """
        Iteratively computes next kappa maps according to the given method
        and the number of iterations.

        Parameters
        ----------
            method : str
                Name of the methode to compute kappa maps.
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
    print("applying {} for {} iteration(s) and {} bpix".format(method, Niter, bpix))
    kE, kB = ks93(g1map * mask, g2map * mask)  # A first estimate of kappa maps
    if method == pm.MethodNames.KS:  # If the desired method is KS...
        return kE, kB                # ... returns the first estimate
    if not dilation:
        constraint = generate_constraint(mask, bpix)
    min_threshold, max_threshold = 0, 0
    if method == pm.MethodNames.DCT:
        # The maximum value of the DCT transform of the E-mode kappa map
        # is used as the maximum threshold value
        max_threshold = np.max(dct2d(kE))
    range_ = tqdm(range(1, Niter)) if tqdm_import else range(1, Niter)
    for i in range_:
        if dilation:
            constraint = generate_constraint(mask, bpix, dilation, i, Niter)
        next_kE, next_kB = next_step_gradient(
            kE, kB, g1map, g2map, mask, reduced)
        kE, kB = projection(method, next_kE, next_kB, i, Niter,
                            max_threshold, min_threshold, constraint, mask)
    return kE, kB


def compute_kappa(config):
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
            Image
                Image with both modes of the computed kappa maps.

    """
    config = copy.deepcopy(config)  # Copy of the configuration
    gammas = Image.from_fits(config['gammas'])  # Loads gamma from fits file
    mask = Image.from_fits(config['mask'])  # Loads mask from fits file
    g1map = gammas.get_layer(0)
    g2map = gammas.get_layer(1)
    # Estimates kappa maps
    kE, kB = iterative(config['method'], g1map, g2map, mask.get_layer(),
                       config['niter'], config['bconstraint'], config['reduced'])
    data = np.array([kE, kB])
    header = config._config
    header.update({'NAME': "computed kappa"})
    return Image(data, header)


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


def bin2d(x, y, v=None, w=None, npix=None, extent=None, verbose=False, timed=False):
    """
        Bin values v according to position (x, y), taking the average of values
        falling into the same bin. Averages are weighted if w is provided. If
        v is not given, return the bin count map.

        Bin edges are computed according to npix such that in each dimension,
        the min (max) position value lies at the center of its first (last) bin.

        Parameters
        ----------
            x, y : array_like
                Position arrays.
            v : array_like, optional
                Values to bin, potentially many arrays of len(x) as [v1, v2, ...].
            w : array_like, optional
                Weight values for v.
            npix : int list as [nx, ny], optional
                If npix = N, use [N, N]. Defaults to [32, 32] if not provided.
            verbose : bool
                If true, print details.
            timed : bool
                If true, print total time taken.

        Returns
        -------
            2d array
                values v binned into pixels.
    """
    start_time = time.time()

    # TODO: verify inputs
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if v is not None:
        v = np.atleast_1d(v)
        if len(v.shape) == 1:
            v = [v]
    if w is not None:
        w = np.atleast_1d(w)
    else:
        w = np.ones_like(x)

    if npix is not None:
        npix = map(int, np.atleast_1d(npix))  # Note: map() returns a list and
        if len(npix) == 2:                    # needs modification for Python3!
            n = npix
        elif len(npix) == 1:
            n = 2 * npix
        else:
            print("Invalid npix. Returning None.")
            return None
    else:
        n = [32, 32]

    # Determine 2D space geometry
    if extent is not None:
        xlow, xhigh, ylow, yhigh = extent
    else:
        ymin, ymax = min(y), max(y)
        xmin, xmax = min(x), max(x)
        halfdx = float(xmax - xmin) / (2 * n[0] - 2)
        halfdy = float(ymax - ymin) / (2 * n[1] - 2)
        xlow = xmin - halfdx
        xhigh = xmax + halfdx
        ylow = ymin - halfdy
        yhigh = ymax + halfdy
    xedges = np.linspace(xlow, xhigh, n[0] + 1)
    yedges = np.linspace(ylow, yhigh, n[1] + 1)

    # For debugging
    if verbose:
        print("xmin, xmax:  {0}, {1}".format(xmin, xmax))
        print("xlow, xhigh: {0}, {1}".format(xlow, xhigh))
        print("xedges: {}".format(xedges))
        print("dx size:   {}".format(halfdx * 2))
        print("ymin, ymax:  {0}, {1}".format(ymin, ymax))
        print("ylow, yhigh: {0}, {1}".format(ylow, yhigh))

    # Do fast binning on 1D arrays
    indx = np.digitize(x, xedges) - 1
    indy = np.digitize(y, yedges) - 1
    size_1d = n[1] * n[0]
    ind_1d = indy * n[0] + indx

    if v is None:
        # Determine pixel counts and return
        nmap = np.bincount(ind_1d, minlength=size_1d)
        if timed:
            print("Time: {0:.3f} s".format(time.time() - start_time))
        return nmap.reshape(n[1], n[0])

    # Weight sums map
    wmap = np.bincount(ind_1d, weights=w, minlength=size_1d)
    # Avoid division by zero in empty pixels
    nmap = np.copy(wmap)  # TODO Can we do this without copying ?
    nmap[wmap == 0] = 1
    # Compute binned v maps
    vmaps = [np.bincount(ind_1d, weights=(v[i] * w), minlength=size_1d) / nmap
             for i in range(len(v))]
    if len(vmaps) == 1:
        binnedmap = np.reshape(vmaps[0], (n[1], n[0]))
    else:
        binnedmap = np.reshape(vmaps, (len(v), n[1], n[0]))

    if timed:
        print("Time: {0:.3f} s".format(time.time() - start_time))

    return binnedmap


def ks93(g1map, g2map):
    """
        Compute Kaiser-Squires inversion mass map from binned gamma maps.

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
            ks93inv : Invers of ks93 function.
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


def ks93inv(kEmap, kBmap):
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
            ks93 : Invers of ks93inv function.
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
    g1map, g2map = ks93inv(k1map, k2map)
    # Apply the mask and noise over the compute gamma maps...
    gamma = Image(np.array([(g1map + n1) * mask, (g2map + n2) * mask]))
    # ... Then it is stored in a fits file under the given name
    gamma.save(g_path)
