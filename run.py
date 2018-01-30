#! /usr/bin/python
# -*- coding: utf-8 -*-

from g2k_lib.projection_methods import HARMONIC, LINEAR, ERF
from g2k_lib.operations import compute_errors, compute_kappa, add_padding
from g2k_lib.objects import Config, Image
from g2k_lib.metrics import get_error
import argparse
import json
import sys
import os


# Below is the parser to handle the input arguments that will be provided
class Parser(object):
    def __init__(self, argv=sys.argv):
        parser = argparse.ArgumentParser(
            description="Determine which action to take.")
        parser.add_argument('action', type=str, choices=['compute', 'evaluate', 'visualize'],
                            help='Action to perform')

        args = parser.parse_args(
            [argv[1] if len(argv) > 1 else argv])
        del argv[1]

        getattr(self, args.action)()

    @classmethod
    def compute(cls, argv=sys.argv[2:]):
        """
        usage: compute [-h] -g GAMMAS -k MASK [-b BCONSTRAINT]
                       [-m {DCT inpainting,full DCT,iKS,iKS Relaxed,KS}] [-n NITER]
                       [-c CONFIG] [-o OUTPUT] [-p] [-r] [-f] [--reduced]

        optional arguments:
          -h, --help            show this help message and exit
          -g GAMMAS, --gammas GAMMAS
                                Path to the fits file containing gamma maps.
          -k MASK, --mask MASK  Path to the fits file containing the mask.
          -b BCONSTRAINT, --bconstraint BCONSTRAINT
                                Number of border constraint pixels (Bzero: for all
                                pixels to zero).
          -m {DCT inpainting,full DCT,iKS,iKS Relaxed,KS}, --method {DCT inpainting,full DCT,iKS,iKS Relaxed,KS}
                                Method to be used in the computation of kappa maps.
          -n NITER, --niter NITER
                                Number of iteration to compute kappa maps.
          -c CONFIG, --config CONFIG
                                Configuration name stored in ../configs/rConfig.json.
          -o OUTPUT, --output OUTPUT
                                Name under the output fits file will be saved.
          -p, --plot            Plot the output (but does not save!!).
          -r, --rename          Enable the auto-renaming for the output files to add
                                more information in file names.
          -f, --force           Overwrite the output file if it already exists
          --reduced             Compute convergence maps using reduced shear maps
                                instead of observed shear maps
        """

        parser = argparse.ArgumentParser(description=__doc__, prog="compute")
        parser.add_argument("--gamma", type=str, required=True,
                            help="Path to the fits file containing gamma maps.")
        parser.add_argument("--mask", type=str,
                            help="Path to the fits file containing the mask.")
        parser.add_argument("-n", "--niter", default=1, type=int,
                            help="Number of iteration to compute kappa maps.")
        parser.add_argument("-b", "--bpix", default="None", type=str,
                            help="Number of border constraint pixels (Bzero: for all pixels to zero).")
        parser.add_argument("--relaxed", action="store_true",
                            help="Enables relaxed border constraint.")
        parser.add_argument("--relax-type", type=str, default=ERF, choices=[
                            LINEAR, ERF, HARMONIC], help="Determines the decreasing law followed by the relaxation parameter, used when --relaxed is given.")
        parser.add_argument("--dct", action="store_true",
                            help="Enables DCT filtering over E-mode.")
        parser.add_argument("--dct-type", type=str, default=ERF, choices=[
                            LINEAR, ERF], help="Determines the decreasing law followed by the threshold parameter, used when --dct is given.")
        parser.add_argument("--dct-block-size", type=int,
                            help="Determines the size of block for DCT computation, used when --dct is given.")
        parser.add_argument("--overlap", action="store_true",
                            help="Enables the overlapping method for DCT computation, used when --dct is given.")
        parser.add_argument("--sbound", action="store_true",
                            help="Enables standard deviation constraint to data located inside mask holes.")
        parser.add_argument("--dilation", action="store_true",
                            help="Enables a linear decreasing of BPIX over iterations.")
        parser.add_argument("--reduced", action="store_true",
                            help="Compute convergence maps using reduced shear maps instead of observed shear maps")
        parser.add_argument("--verbose", action="store_true",
                            help="Enables information display.")
        parser.add_argument("--no-padding", action="store_true",
                            help="Disables padding option.")

        parser.add_argument("--output", type=str,
                            help="Output file name.")
        parser.add_argument("--plot", action="store_true",
                            help="Plot the output (but does not save!!).")
        parser.add_argument("--rename", action="store_true",
                            help="Enable the auto-renaming for the output files to add more information in file names.")
        parser.add_argument("-f", "--force", action="store_true",
                            help="Overwrite the output file if it already exists")

        args = parser.parse_args(argv)

        gamma_path = os.path.abspath(args.gamma)
        mask_path = os.path.abspath(args.mask) if args.mask else None
        niter = args.niter
        bpix = args.bpix
        relaxed = args.relaxed
        relax_type = args.relax_type
        dct = args.dct
        dct_type = args.dct_type
        dct_block_size = args.dct_block_size
        overlap = args.overlap
        sbound = args.sbound
        dilation = args.dilation
        reduced = args.reduced
        verbose = args.verbose
        no_padding = args.no_padding
        output = args.output
        plot = args.plot
        rename = args.rename
        force = args.force

        # Input control
#=============================================================================
        if not os.path.exists(gamma_path):
            sys.exit("File '{}' does not exist".format(gamma_path))
        if mask_path and not os.path.exists(mask_path):
            sys.exit("File '{}' does not exist".format(mask_path))
        if niter < 0:
            print("WARNING: Negative iteration number. Won't perform any iteration.")
        if bpix not in {"None", "Bzero"}:
            try:
                int(bpix)
            except:
                sys.exit("bpix must be 'None' (default), 'Bzero' or an integer.")
        if not output:
            print("The output name field is empty, the result won't be saved")
        else:
            if output[-5:] != ".fits":
                output += ".fits"
            if rename:
                _dct = "DCT" if dct else ""
                _relaxed = "relaxed" if relaxed else ""
                _relax_type = relax_type if relaxed and relax_type else ""
                _dct_type = dct_type if dct and dct_type else ""
                _dct_block_size = str(
                    dct_block_size) if dct and dct_block_size else ""
                _overlap = "overlap" if dct and overlap else ""
                _sbound = "sbound" if sbound else ""
                _dilation = "dilation" if dilation else ""
                _reduced = "reduced" if reduced else ""
                _niter = "{}iter".format(niter)
                _bpix = "{}bpix".format(bpix)
                output = output.replace(
                    ".fits", "_" + str.join('_', filter(None, [_niter, _bpix, _reduced, _dct, _dct_type, _dct_block_size,
                                                               _overlap, _dilation, _relaxed, _relax_type, _sbound])) + ".fits")
            if os.path.exists(output) and not force:
                sys.exit(
                    "The output file already exists: '{}' please use -f/--force option to overwrite".format(os.path.abspath(output)))
#=============================================================================

        kappa = compute_kappa(gamma, mask, niter, bpix, relaxed, relax_type, dct, dct_type,
                              dct_block_size, overlap, sbound, reduced, dilation, verbose, no_padding)

        if output:
            if os.path.exists(output) and force:
                os.remove(output)
            kappa.save(output)
        if plot:
            kappa.plot()
            raw_input()

    def evaluate(self, argv=sys.argv[2:]):
        """
        usage: run.py [-h] -g GND_TRUTH -k KAPPAS -o OUTPUT [-p] [--config CONFIG]

        Compute errors between two fits.

        optional arguments:
          -h, --help            show this help message and exit
          -g GND_TRUTH, --gnd_truth GND_TRUTH
                                Path to the kappas ground truth.
          -k KAPPAS, --kappas KAPPAS
                                Path to the computed kappas
          -o OUTPUT, --output OUTPUT
                                File where to store the result value. Will add the
                                value to the provided JSON file, will create one if
                                the file does not already exists.
          -p, --plot            Plot the difference maps.
          --config CONFIG       Configuration name stored in rConfigs.json.
        """
        parser = argparse.ArgumentParser(
            description='Compute errors between two fits.')
        parser.add_argument('-g', '--gnd_truth', type=str, required=True,
                            help='Path to the kappas ground truth.')
        parser.add_argument('-k', '--kappas', type=str, required=True,
                            help='Path to the computed kappas')
        parser.add_argument('-o', '--output', type=str, required=True,
                            help='File where to store the result value.\n Will add the value to the provided JSON file, will create one if the file does not already exists.')
        parser.add_argument('-p', '--plot', action='store_true',
                            help='Plot the difference maps.')
        parser.add_argument('--config', type=str, default='default',
                            help='Configuration name stored in rConfigs.json.')
        args = parser.parse_args(argv)
        config = Config.get_configuration('rConfigs', args)

        diff = compute_errors(
            config.kappas, config.gnd_truth, config.output)

        if config.plot:
            diff.plot()
            raw_input()

    def visualize(self, argv=sys.argv[2:]):
        """
        usage: run.py [-h] -r REGISTER_PATH [--config CONFIG]

        Plot data from result files.

        optional arguments:
          -h, --help            show this help message and exit
          -r REGISTER_PATH, --register-path REGISTER_PATH
                                Path to the result register JSON file.
          --config CONFIG       Configuration name stored in vConfigs.json.
        """
        parser = argparse.ArgumentParser(
            description='Plot data from result files.')
        parser.add_argument('-r', '--register-path', type=str, help='Path to the result register JSON file.',
                            required=True)
        parser.add_argument('--config', type=str, default='default',
                            help='Configuration name stored in vConfigs.json.')
        args = parser.parse_args(argv)
        config = Config.get_configuration('vConfigs', args)

        if not os.path.exists(config.register_path):
            sys.exit("The specified register file does't exist: '{}'".format(
                config.register))

        visualize(config)


if __name__ == '__main__':
    Parser()
