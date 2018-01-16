__g2k_lib__  v0.0.1
=======================

_Written by Remy Leroy_

This module aims to provide several methods to compute iteratively convergence maps from shear maps.

------------
Installation
------------

Type `sudo python setup.py install` in a shell at the root directory.

--------
Features
--------

* Computation of convergence maps by using the `compute` action.
* Evaluation of the error with respect to a reference convergence map by using the `evaluate` action.
* Plot curves using the `visualize` action.

-------
Example
-------

The command syntax is the following:

```
    ./run.py compute --gamma ./shear.fits --mask ./mask.fits
```
The above command makes the computation using the shear maps stored in `./shear.fits` and the mask stored in `./mask.fits` according to the default configuration stored in the `./configs/rConfigs.json`. You can add your own configuration to this file and call it using the `--config yourconfigname` option.  

The result won't be saved, add `--output ./output.fits` to store the result into the `./output.fits` file.  
Add `-p` to plot the result.

Refer to the `./run.py compute --help` for more information.  
