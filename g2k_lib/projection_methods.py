# -*- coding: utf-8 -*-

from dct_interpolation import filtering, get_threshold_n_erf


class MethodNames():
    """
        This class gathers all the available method names.

    """
    KS = "KS"
    """ str: Value relative to the single Kaiser-Squires method."""
    IKS = "iKS"
    """ str: Value relative to the iterative Kaiser-Squires method."""
    IKS_RELAXED = "iKS Relaxed"
    """ str: Value relative to the iterative and relaxed Kaiser-Squires method."""
    DCT = "DCT inpainting"
    """ str: Value relative to the DCT inpainting method."""
    FULL_DCT = "full DCT"

    @classmethod
    def get_names(cls):
        """
            Returns the list of all the available method names.
        """
        return [getattr(cls, a) for a in dir(cls) if "__" not in a and type(getattr(cls, a)) is str]


def iks_relaxed(next_kE, next_kB, i, Niter, constraint, mask):
    # Decreasing weight according to the number of iterations
    w = float(Niter) - i
    kE = next_kE
    # Apply the relaxed constraint to the B mode
    relaxed_constraint = (constraint + (1 - constraint) / float(w)) * mask
    kB = next_kB * relaxed_constraint
    return kE, kB


def iks(next_kE, next_kB, constraint):
    kE = next_kE
    kB = next_kB * constraint  # Apply the relaxed constraint to the B mode
    return kE, kB


def dct_inpaint(next_kE, next_kB, i, Niter, max_threshold, min_threshold, constraint, mask):
    # Evaluates thresholding value for DCT filtering
    threshold = get_threshold_n_erf(n=i, Niter=Niter - 1,
                                    th_max=max_threshold,
                                    th_min=min_threshold)
    # DCT filtering
    next_kE = filtering(next_kE, threshold)
    # Constraint over the standard deviation value inside holes in the mask
    std_out = next_kE[mask.astype(bool)].std()
    std_in = next_kE[~mask.astype(bool)].std()
    next_kE *= mask + (~mask.astype(bool)).astype(int) * (std_out / std_in)

    # Combination of the DCT by the relaxed iterative KS constraint
    kE, kB = iks_relaxed(next_kE, next_kB, i, Niter, constraint, mask)

    return kE, kB


def full_dct(next_kE, next_kB, i, Niter, max_threshold, min_threshold, constraint, mask):
    # Evaluates thresholding value for DCT filtering
    threshold = get_threshold_n_erf(n=i, Niter=Niter - 1,
                                    th_max=max_threshold,
                                    th_min=min_threshold)
    # DCT filtering
    next_kE = filtering(next_kE, threshold)
    # Constraint over the standard deviation value inside holes in the mask
    std_out = next_kE[mask.astype(bool)].std()
    std_in = next_kE[~mask.astype(bool)].std()
    next_kE *= mask + (~mask.astype(bool)).astype(int) * (std_out / std_in)

    # kill the B mode
    kE, kB = iks(next_kE, next_kB, 0)

    return kE, kB
