# -*- coding: utf-8 -*-

from astropy.io import fits
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os


class Image(object):
    """
        Object storing array_like data as images

        Attributes
        ----------
            data : array_like
                Image data
            header : dict
                Image meta-data
            file : file
                Image file
            filename : str
                Image file name
            layers : int
                Number of layers into the image file

    """

    def __init__(self, data, header={}, filename=''):
        self.data = data
        if len(self.data.shape) == 2:  # if the dimensionality of the image is greater than 1
            self.data = np.array([data])
        self.header = header
        self.file = None
        self.filename = filename

    @classmethod
    def from_fits(cls, filename):
        """
            Create an Image object from data contained into a fits file
            by providing the file's path, adding a NAME field into the header.
        """
        if type(filename) is not str:
            raise TypeError(
                "from_fits() argument must be a string, not '{}'.".format(type(filename)))
        if not os.path.exists(filename):
            sys.exit("File '{}' does not exist".format(filename))
        f = fits.open(filename, mode='update')
        header = f[0].header
        if 'NAME' not in header.keys():
            header['NAME'] = os.path.splitext(os.path.basename(filename))[0]
        data = f[0].data

        image = cls(data, header, filename)
        image.file = f

        return image

    def save(self, filename=None, overwrite=False):
        """
            Save the image into a fits file under the provided file name.
        """
        if filename:
            if os.path.exists(filename):
                if not overwrite:
                    permission = raw_input(
                        "File '" + filename + "' already exists. Do you want to overwrite it ? (y/[n]):")
                    overwrite = True if permission and permission in {
                        'y', 'Y'} else False
                if overwrite:
                    self.filename = filename
                else:
                    sys.exit("Save action canceled.")
            self.filename = filename
        elif not self.filename:
            sys.exit("File name not found. Cannot save the image.")
        if self.layers == 1:
            data = self.data[0]
        else:
            data = self.data

        if not os.path.exists(os.path.dirname(self.filename)):
            os.system('mkdir -p {}'.format(os.path.dirname(self.filename)))
        if self.file:
            self.file.close()
        fits.writeto(self.filename, data)
        self.file = fits.open(self.filename, mode='update')
        self.file[0].header.update(self.header)
        self.file[0].data = data
        self.file.flush()

    @property
    def layers(self):
        """
            Dimensionality of the image
        """
        return self.data.shape[0]

    def get_layer(self, layer=0):
        """
            Returns the image at the given layer (default layer=0)
        """
        return self.data[layer]

    def plot(self, layer=-1, filename=None, **prefs):
        cmap = prefs['cmap'] if prefs.has_key("cmap") else 'gist_stern'
        xlabel = prefs['xlabel'] if prefs.has_key("xlabel") else ''
        ylabel = prefs['ylabel'] if prefs.has_key("ylabel") else ''
        title = prefs['title'] if prefs.has_key("title") else ''
        clabel = prefs['clabel'] if prefs.has_key("clabel") else ''

        number = 1 if layer >= 0 else self.layers
        f, axis = plt.subplots(1, number)
        plt.title(title)
        if number == 1:
            axis = np.array([axis])
        for l in xrange(len(axis)):
            vmin = prefs['vmin'] if prefs.has_key(
                "vmin") else self.data[l].min()
            vmax = prefs['vmax'] if prefs.has_key(
                "vmax") else self.data[l].max()
            im = axis[l].imshow(self.data[l], cmap=cmap,
                                origin='lower', vmin=vmin, vmax=vmax)
            axis[l].set_xlabel(xlabel)
            axis[l].set_ylabel(ylabel)

            f.colorbar(im, ax=axis[l], label=clabel)
        if filename:
            plt.savefig(filename)
        else:
            plt.show(block=False)
