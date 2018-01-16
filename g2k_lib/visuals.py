# -*- coding: utf-8 -*-

from objects import ResultRegister, Counter
from projection_methods import MethodNames as mn
import matplotlib.pyplot as plt
import numpy as np
import copy


def visualize(config):
    """
    This function constructs and displays plots according to the given coniguration
    """
    register = ResultRegister(config["register_path"])
    attributes = ["niter", "bpix"]
    plt.figure(Counter().count)
    for method in config.methods:  # Iterates over methods we want to display
        # If there exist some entries for the given method...
        if method in register.data.keys():
            # Store the entries related to the method into a hash table ndata
            ndata = register.data[method]

            # Bzero -> -1
            # None -> -2

            def get_bpix(bpix):
                if bpix == "Bzero":
                    return -1
                elif bpix == "None":
                    return -2
                else:
                    return int(bpix)

            # Convert the ndata hash table into a numpy.rec.array data
            ndata = [tuple([int(niter), get_bpix(biter), biterdata['e'], biterdata['b']])
                     for niter, niterdata in ndata.iteritems() for biter, biterdata in niterdata.iteritems()]
            data = np.rec.array(ndata, dtype=[("niter", ">i4"), ("bpix", ">i4"), ("error_e", ">f8"),
                                              ("error_b", ">f8")])

            filtered_data = copy.deepcopy(data)
            unfiltered_data = copy.deepcopy(data)

            # Filters data entries to match with the given configuration
            for attribute in attributes:
                value = getattr(config, attribute)
                if value == "variable":
                    variable = attribute
                    # if not (method == mn.DCT and variable == "bpix"):
                    data = data[(data[attribute] >= config.min) &  # Keeps data where de variable attribute
                                (data[attribute] <= config.max)]  # lies within the configuration range values
                else:
                    data = data[data[attribute] == value]
                    filtered_data = filtered_data[filtered_data[attribute] == value]

            data.sort(order=variable)

            error_name = "error_{}".format(config.mode)

            # Adds curves of the corresponding method to the plot
            # if method == mn.DCT and variable == "bpix":
            #     p = plt.plot([config.min, config.max], [data[0][error_name], data[0][error_name]], label=method)
            if method == mn.KS:
                p = plt.plot([config.min, config.max], [unfiltered_data[(unfiltered_data["niter"] == 1) &
                                                                        (unfiltered_data["bpix"] == -2)][error_name]] * 2, "k-.", label=method)
            else:
                p = plt.plot(data[variable], data[error_name],
                             label=method + " Bborder")

            # To plot a line corresponding to a reference value
            for ref in config.refs:
                xs = []
                ys = []
                for x in data[variable]:
                    if x >= 0:
                        if ref == "Bzero":
                            v = filtered_data[filtered_data["bpix"]
                                              == -1][error_name]
                            if len(v) > 0:
                                xs.append(x)
                                ys.append(v)
                if len(xs) > 0:
                    plt.plot(xs, ys, '--', label="{} {}".format(method,
                                                                ref), c=p[0].get_color())
    plt.ylim(0)
    plt.legend()
    plt.xlabel(config.xlabel)
    plt.ylabel(config.ylabel)
    plt.title(config.title)
    plt.show(block=False)
    raw_input()
