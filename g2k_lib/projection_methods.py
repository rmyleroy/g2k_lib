# -*- coding: utf-8 -*-

from transforms import filtering, get_threshold_n_erf, get_threshold_n_lin

HARMONIC = "harm"
LINEAR = "lin"
ERF = "erf"


def iks_relaxed(kB, i, Niter, constraint, mask, type_=LINEAR, verbose=False):
    # Decreasing weight according to the number of iterations
    if type_ == LINEAR:
        weight = get_threshold_n_lin(n=i, Niter=Niter, th_max=0, th_min=1)
    elif type_ == ERF:
        weight = get_threshold_n_erf(n=i, Niter=Niter, th_max=0, th_min=1)
    elif type_ == HARMONIC:
        weight = 1 / (float(Niter + 1) - i)
    print("weight: {:.4}".format(weight)) if verbose else None
    # Apply the relaxed constraint to the B mode
    relaxed_constraint = (constraint + (1 - constraint) * weight) * mask
    next_kB = kB * relaxed_constraint
    return next_kB


def iks(kB, constraint):
    next_kB = kB * constraint  # Apply the relaxed constraint to the B mode
    return next_kB


def dct_inpaint(kE, i, Niter, max_threshold, min_threshold, type_=ERF, verbose=False):
    # Evaluates threshold value for DCT filtering
    threshold = get_threshold_n_erf(n=i, Niter=Niter,
                                    th_max=max_threshold,
                                    th_min=min_threshold)
    print("threshold: {:.4}".format(threshold)) if verbose else None
    # DCT filtering
    next_kE = filtering(kE, threshold)

    return next_kE


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
