__g2k_lib__
===========


| **Author**  | **Remy Leroy** |
|:-----------:|:--------------:|
| **Version** |   **0.0.1**    |
|  **Date**   | **Jan. 2018**  |

This module aims to provide several methods to compute iteratively convergence maps from shear maps under FITS format.

-------
Content
-------
```
.
├── configs
│   ├── rConfigs.json
│   └── vConfigs.json
├── g2k_lib
│   ├── dct_interpolation.py
│   ├── __init__.py
│   ├── metrics.py
│   ├── objects.py
│   ├── operations.py
│   ├── projection_methods.py
│   ├── visuals.py
├── MANIFEST.in
├── README.md
├── requirements.txt
├── run.py
└── setup.py
```

------------
Installation
------------

You can either install it as a regular package by typing
```sudo python setup.py install``` in a shell

**OR**

just install all the dependencies listed in `requirements.txt` using the following command: ```sudo pip install -r requirements.txt```

--------
Features
--------

This package contains an executable pyton file `run.py` that allows the user to perform several actions which are:
* Computation of convergence maps by calling the `compute` action.
* Evaluation of the error with respect to a reference convergence map by calling the `evaluate` action.
* Plot error curves by calling the `visualize` action.

A call is done as follows: ```./run.py <action> <parameters>```.
Each action takes its owns set of parameters.

Use ```./run.py <action> --help``` to get a list of all the parameters the action can take.




-------
Example
-------

Let's suppose the data to be processed are located in the `./data` folder as follows:
```
.
├── configs/
│   ├── rConfigs.json
│   └── vConfigs.json
|
├── data/
│   ├── convergence_maps.fits
│   ├── mask.fits
│   └── shear_maps.fits
|
├── g2k_lib/
│   ├── dct_interpolation.py
│   ├── __init__.py
│   ├── metrics.py
│   ├── objects.py
│   ├── operations.py
│   ├── projection_methods.py
│   ├── visuals.py
├── MANIFEST.in
├── README.md
├── requirements.txt
├── run.py
└── setup.py
```
Where `mask.fits` is a binary map of missing data; `shear_maps.fits` consists in two shear maps (E-mode map and B-mode map); `convergence_maps.fits` is a simulated convergence map corresponding to `shear_maps.fits` that can be used as a ground truth map to evaluate the quality of reconstruction.

`compute` action
----------------
```
./run.py compute --gamma ./data/shear_maps.fits  --mask ./data/mask.fits --method 'iKS' --niter 50 --bconstraint 10 --output ./data/result.fits
```
The above instruction will compute the convergence map corresponding to the given shear map, mask map and the method of computation; Here the result will be computed using the iterative Kaiser-Squires method for 50 iterations and with 10 pixels for border constraint and it will be stored into the `result.fits` file.

`result.fits` will contain two maps, one for each mode.


`evaluate` action
-----------------

To evaluate the quality of the reconstruction, it is possible to compare `result.fits` with `convergence_maps.fits`, the `evaluate` action should be used to do so.

```
./run.py evaluate --kappa ./data/result.fits --gnd_truth ./data/convergence_maps.fits --output ./data/table.json
```
The error value will be stored into a JSON file corresponding to the method used to compute the estimated convergence map.
