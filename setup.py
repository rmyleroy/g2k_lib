# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import g2k_lib

setup(
    name='g2k_lib',
    version='1.0.0',
    description='A module to compute convergence maps using several methods',
    long_description=open("README.md").read(),
    author='Remy Leroy',
    author_email='rmyleroy@gmail.com',
    packages=find_packages(),  # same as name
    install_requires=['tqdm', 'astropy', 'matplotlib',
                      'numpy', 'scipy'],  # external packages as dependencies
)
