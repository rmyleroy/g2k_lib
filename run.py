#! /usr/bin/python
# -*- coding: utf-8 -*-

from g2k_lib.projection_methods import MethodNames
from g2k_lib.operations import compute_errors, compute_kappa
from g2k_lib.visuals import visualize
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

        parser = argparse.ArgumentParser(description=__doc__, prog='compute')
        parser.add_argument('-g', '--gammas', type=str, required=True,
                            help='Path to the fits file containing gamma maps.')
        parser.add_argument('-k', '--mask', type=str, required=True,
                            help='Path to the fits file containing the mask.')
        parser.add_argument('-b', '--bconstraint', type=str,
                            help='Number of border constraint pixels (Bzero: for all pixels to zero).')
        parser.add_argument('-m', '--method', type=str, choices=MethodNames.get_names(),
                            help='Method to be used in the computation of kappa maps.')
        parser.add_argument('-n', '--niter', type=int,
                            help='Number of iteration to compute kappa maps.')

        parser.add_argument('-c', '--config', type=str, default='default',
                            help='Configuration name stored in ../configs/rConfig.json.')
        parser.add_argument('-o', '--output', type=str,
                            help='Name under the output fits file will be saved.')
        parser.add_argument('-p', '--plot', action='store_true',
                            help='Plot the output (but does not save!!).')
        parser.add_argument('-r', '--rename', action='store_true',
                            help='Enable the auto-renaming for the output files to add more information in file names.')
        parser.add_argument('-f', '--force', action='store_true',
                            help='Overwrite the output file if it already exists')
        parser.add_argument('--reduced', action='store_true',
                            help='Compute convergence maps using reduced shear maps instead of observed shear maps')

        args = parser.parse_args(argv)
        config = Config.get_configuration('rConfigs', args)

        if config.rename:
            config.output = config.output.replace('.fits', '_{}_{}iter_{}bpix.fits'.format(config.method,
                                                                                           config.niter, config.bconstraint))

        if os.path.exists(config.output) and not config.force:
            sys.exit(
                "The output file already exists: '{}' please use -f option to overwrite".format(config.output))
        if not config.output:
            print("The output name field is empty, the result won't be saved")

        kappa = compute_kappa(config)

        # According to the -o option, saves or plots both kappaE and kappaB.
        if config.output:
            if os.path.exists(config.output):
                os.remove(config.output)
            kappa.save(config.output)
        if config.plot:
            kappa.plot()
            raw_input()

    def evaluate(self, argv=sys.argv[2:]):
        parser = argparse.ArgumentParser(
            description='Compute errors between two fits.')
        parser.add_argument('-g', '--gnd_truth', type=str, required=True,
                            help='Path to the kappas ground truth.')
        parser.add_argument('-k', '--kappas', type=str, required=True,
                            help='Path to the computed kappas')
        parser.add_argument('-o', '--output', type=str, required=True,
                            help='File where to store the result value.\n Will add the value to the provided JSON file, will create one if the file does not already exists.')
        parser.add_argument('-p', '--plot', action='store_true',
                            help='Plot the difference E-mode.')
        parser.add_argument('--config', type=str, default='default',
                            help='Configuration name stored in config.json.')
        args = parser.parse_args(argv)
        config = Config.get_configuration('rConfigs', args)

        diff = compute_errors(
            config.kappas, config.gnd_truth, config.output)

        if config.plot:
            diff.plot()
            raw_input()

    def visualize(self, argv=sys.argv[2:]):
        parser = argparse.ArgumentParser(
            description='Plot data from result files.')
        parser.add_argument('-r', '--register-path', type=str, help='Path to the result register json file.',
                            required=True)
        parser.add_argument('--config', type=str, default='default',
                            help='Configuration name stored in vConfig.json.')
        args = parser.parse_args(argv)
        config = Config.get_configuration('vConfigs', args)

        if not os.path.exists(config.register_path):
            sys.exit("The specified register file does't exist: '{}'".format(
                config.register))

        visualize(config)


if __name__ == '__main__':
    Parser()
