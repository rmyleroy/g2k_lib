#! /usr/bin/python
# -*- coding: utf-8 -*-

from g2k_lib.projection_methods import HARMONIC, LINEAR, ERF
from g2k_lib.operations import compute_errors, compute_kappa
from g2k_lib.objects import Config
import argparse
import json
import sys
import os


# Below is the parser to handle the input arguments that will be provided
class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Determine which action to take.")
        parser.add_argument('action', type=str, choices=['compute', 'evaluate', 'visualize'],
                            help='Action to perform')

        args = parser.parse_args(
            [sys.argv[1] if len(sys.argv) > 1 else sys.argv])
        del sys.argv[1]

        getattr(self, args.action)()

    def compute(self, argv=sys.argv[2:]):
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
        parser.add_argument("--gammas", type=str, required=True,
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
                            help="Name under the output fits file will be saved.")
        parser.add_argument("--plot", action="store_true",
                            help="Plot the output (but does not save!!).")
        parser.add_argument("--rename", action="store_true",
                            help="Enable the auto-renaming for the output files to add more information in file names.")
        parser.add_argument("-f", "--force", action="store_true",
                            help="Overwrite the output file if it already exists")

        args = parser.parse_args(argv)

        if not args.output:
            print("The output name field is empty, the result won't be saved")
        else:
            output = args.output
            if output[-5:] != ".fits":
                output += ".fits"
            if args.rename:
                _dct = "_DCT" if args.dct else ""
                _relaxed = "_relaxed" if args.relaxed else ""
                _relax_type = "_{}".format(
                    args.relax_type) if args.relaxed and args.relax_type else ""
                _dct_type = "_{}".format(
                    args.dct_type) if args.dct and args.dct_type else ""
                _sbound = "_sbound" if args.sbound else ""
                _dilation = "_dilation" if args.dilation else ""
                _reduced = "_reduced" if args.reduced else ""
                _niter = "_{}iter".format(args.niter)
                _bpix = "_{}bpix".format(args.bpix)
                output = output.replace(
                    ".fits", _niter + _bpix + _reduced + _dct + _dct_type + _dilation + _relaxed + _relax_type + _sbound + ".fits")
            if os.path.exists(output) and not args.force:
                sys.exit(
                    "The output file already exists: '{}' please use -f option to overwrite".format(os.path.abspath(args.output)))

        # kappa = compute_kappa(args.gammas, agrs.mask, )
        return
        # According to the -o option, saves or plots both kappaE and kappaB.
        if output:
            if os.path.exists(output) and args.force:
                os.remove(output)
            kappa.save(output)
        if args.plot:
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
