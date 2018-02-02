# -*- coding: utf-8 -*-

from transforms import filtering, get_threshold_n_erf, get_threshold_n_lin

HARMONIC = "harm"
LINEAR = "lin"
ERF = "erf"


def iks_relaxed(kB, i, Niter, constraint, mask, relax_type, verbose=False):
    # Decreasing weight according to the number of iterations
    if relax_type == LINEAR:
        weight = get_threshold_n_lin(n=i, Niter=Niter, th_max=0, th_min=1)
    elif relax_type == ERF:
        weight = get_threshold_n_erf(n=i, Niter=Niter, th_max=0, th_min=1)
    elif relax_type == HARMONIC:
        weight = 1 / (float(Niter + 1) - i)
    if verbose:
        print("weight: {:.4}".format(weight))
    # Apply the relaxed constraint to the B mode
    relaxed_constraint = (constraint + (1 - constraint) * weight) * mask
    next_kB = kB * relaxed_constraint
    return next_kB


def iks(kB, constraint):
    next_kB = kB * constraint  # Apply the relaxed constraint to the B mode
    return next_kB


def dct_inpaint(kE, i, Niter, max_threshold, min_threshold, threshold_type=ERF, block_size=None, overlap=False, verbose=False):
    # Evaluates threshold value for DCT filtering
    if threshold_type == ERF:
        threshold = get_threshold_n_erf(n=i, Niter=Niter,
                                        th_max=max_threshold,
                                        th_min=min_threshold)
    elif threshold_type == LINEAR:
        threshold = get_threshold_n_lin(n=i, Niter=Niter,
                                        th_max=max_threshold,
                                        th_min=min_threshold)
    if verbose:
        print("threshold: {:.4}".format(threshold))
    # DCT filtering
    next_kE = filtering(kE, threshold, block_size, overlap)

    return next_kE
