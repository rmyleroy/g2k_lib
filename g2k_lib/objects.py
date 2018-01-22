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
        f = fits.open(filename, mode='update')
        header = f[0].header
        if 'NAME' not in header.keys():
            header['NAME'] = os.path.splitext(os.path.basename(filename))[0]
        data = f[0].data

        image = cls(data, header, filename)
        image.filename = filename
        image.file = f

        return image

    def save(self, filename=None):
        """
            Save the image into a fits file under the provided file name.
        """
        self.filename = filename
        if self.layers == 1:
            data = self.data[0]
        else:
            data = self.data
        if filename:
            if not os.path.exists(os.path.dirname(filename)):
                os.system('mkdir -p {}'.format(os.path.dirname(filename)))
            if self.file:
                self.file.close()
            fits.writeto(filename, data)
            self.file = fits.open(filename, mode='update')
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
        nbound = prefs['nbound'] if prefs.has_key("nbound") else 256
        ncolors = prefs['ncolors'] if prefs.has_key("ncolors") else 256

        number = 1 if layer >= 0 else self.layers
        f, axis = plt.subplots(1, number)
        plt.title(title)
        if number == 1:
            axis = np.array([axis])
        for l in xrange(len(axis)):
            vmin = prefs['vmin'] if prefs.has_key(
                "vmin") else int(self.data[l].min())
            vmax = prefs['vmax'] if prefs.has_key(
                "vmax") else np.floor(self.data[l].max())
            im = axis[l].imshow(self.data[l], cmap=cmap,
                                origin='lower', vmin=vmin, vmax=vmax)
            axis[l].set_xlabel(xlabel)
            axis[l].set_ylabel(ylabel)

            f.colorbar(im, ax=axis[l], label=clabel)
        if filename:
            plt.savefig(filename)
        else:
            plt.show(block=False)


class ComputedKappa(Image):
    """
        Attributes
        ----------
            method : str
                Name of the method used to compute kappas, stored in the header
            niter : int
                Number of iterations to compute kappas, stored in the header
            bconstraint : int or str
                Type of constraint applied over B component
            mask_path : str
                Path to the mask image used to compute kappas
    """

    def __init__(self, data, header, filename=''):
        super(ComputedKappa, self).__init__(data, header, filename)

    @property
    def method(self):
        return self.header['method']

    @property
    def niter(self):
        return self.header['niter']

    @property
    def bconstraint(self):
        return self.header['bconstraint']

    @property
    def mask_path(self):
        return self.header['mask']


class Config(object):
    """
        Attributes:
            CONF_PATH : str
                Directory where configuration files MUST be stored
            DEFAULT_CONF_NAME : str
                Default configuration name used when omitted
            filename : str
                Name of the configuration file stored in the configuration directory
            config : dict
                Set of (key, value(s))
    """
    CONF_PATH = './configs/'
    DEFAULT_CONF_NAME = 'default'

    def __init__(self, filename, **kwargs):
        self._config = self.__class__._read_json(
            filename, self.__class__.DEFAULT_CONF_NAME)
        self._config.update(kwargs)

        for k, v in self._config.iteritems():
            setattr(self, k, v)

    @classmethod
    def _read_json(cls, filename, config_name):
        """
        Check the existence of the given configuration name into the configuration file
        """
        filename = os.path.join(cls.CONF_PATH, filename) + ".json"
        configs = json.load(open(filename))
        if config_name not in configs.keys():
            sys.exit("{} is not available as a valid configuration in {}".format(
                config_name, filename))
        return configs[config_name]

    @classmethod
    def from_json(cls, filename, config_name):
        """
        Returns the keys and values relative to the given configuration name
        """
        config = cls._read_json(filename, config_name)
        return cls(filename, **config)

    @classmethod
    def get_configuration(cls, filename, args):
        c = Config.from_json(filename, args.config)
        attributes = [attribute for attribute in dir(
            args) if attribute[0] != '_']
        for attribute in attributes:
            if getattr(args, attribute) is not None or not hasattr(c, attribute):
                c._config[attribute] = getattr(args, attribute)
                setattr(c, attribute, getattr(args, attribute))
        return c

    def __getitem__(self, key):
        return self._config[key]

    def __str__(self):
        return "Config({})".format(', '.join(["{}={}".format(k, v) for k, v in self._config.iteritems()]))


class ResultRegister(object):
    """
        Attributes:
            filename : str
                Name of the file where all the results are stored
            data : dict
                Dictionary containing all the errors computed
    """

    def __init__(self, filename):
        self.filename = filename
        try:
            self.data = json.load(open(self.filename))
        except Exception:
            self.data = {}

    def get_error(self, method, niter, bconstraint, mode='e'):
        """
        Returns the errors relative to the given configuration
        """
        try:
            return self.data[method][niter][bconstraint]
        except Exception:
            return None

    def set_error(self, method, niter, bconstraint, error, mode='e'):
        """
        Create an entry to store the error value, overwrite the entry if already existing
        """
        niter = str(niter)
        bconstraint = str(bconstraint)
        if method not in self.data.keys():
            self.data[method] = {}
        if niter not in self.data[method].keys():
            self.data[method][niter] = {}
        if bconstraint not in self.data[method][niter].keys():
            self.data[method][niter][bconstraint] = {}
        self.data[method][niter][bconstraint][mode] = error

    def save(self):
        """
        Store all the values into its file
        """
        try:
            os.makedirs(os.path.dirname(self.filename))
        except OSError as exc:
            pass
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=4, separators=(',', ': '))


class Counter(object):
    """
    This is just a counter called when a new plot figure is initialised
    """

    count = 0

    def __init__(self):
        self.count = self.__class__.count
        Counter.count += 1
