"""Base classes for EnergyFlow."""
from __future__ import absolute_import, division

from abc import ABCMeta, abstractmethod, abstractproperty
import multiprocessing
import sys
import warnings

import numpy as np
from six import with_metaclass

from energyflow.measure import Measure, measure_kwargs
from energyflow.utils import kwargs_check


###############################################################################
# EFBase
###############################################################################
class EFBase(with_metaclass(ABCMeta, object)):

    """A base class for EnergyFlow objects that holds a `Measure`."""

    def __init__(self, kwargs):

        kwargs_check('EFBase', kwargs, allowed=measure_kwargs)
        self._measure = Measure(kwargs.pop('measure'), **kwargs)

    def has_measure(self):
        return hasattr(self, '_measure') and isinstance(self._measure, Measure)

    @property
    def measure(self):
        return self._measure.measure if self.has_measure() else None

    @property
    def beta(self):
        return self._measure.beta if self.has_measure() else None

    @property
    def kappa(self):
        return self._measure.kappa if self.has_measure() else None

    @property
    def normed(self):
        return self._measure.normed if self.has_measure() else None

    @property
    def coords(self):
        return self._measure.coords if self.has_measure() else None

    @property
    def check_input(self):
        return self._measure.check_input if self.has_measure() else None

    @property
    def subslicing(self):
        return self._measure.subslicing if self.has_measure() else None

    def batch_compute(self, events, n_jobs=-1):
        """Computes the value of the EFP on several events.

        **Arguments**

        - **events** : array_like or `fastjet.PseudoJet`
            - The events as an array of arrays of particles in coordinates
            matching those anticipated by `coords`.
        - **n_jobs** : _int_ 
            - The number of worker processes to use. A value of `-1` will attempt
            to use as many processes as there are CPUs on the machine.

        **Returns**

        - _1-d numpy.ndarray_
            - A vector of the EFP value for each event.
        """

        if n_jobs == -1:
            try:
                n_jobs = multiprocessing.cpu_count()
            except:
                n_jobs = 4 # choose reasonable value

        self.n_jobs = n_jobs

        # don't bother setting up a Pool
        if self.n_jobs == 1:
            return np.asarray(list(map(self._batch_compute_func, events)))

        # setup processor pool
        chunksize = min(max(len(events)//self.n_jobs, 1), 10000)
        if sys.version_info[0] == 3:
            with multiprocessing.Pool(self.n_jobs) as pool:
                results = np.asarray(list(pool.imap(self._batch_compute_func, events, chunksize)))
                
        # Pool is not a context manager in python 2
        else:
            pool = multiprocessing.Pool(self.n_jobs)
            results = np.asarray(list(pool.imap(self._batch_compute_func, events, chunksize)))
            pool.close()

        return results


###############################################################################
# EFPBase
###############################################################################
class EFPBase(EFBase):

    def __init__(self, kwargs):

        # initialize base class if measure needed
        if not kwargs.pop('no_measure', False):

            # set default measure for EFPs
            kwargs.setdefault('measure', 'hadr')
            super(EFPBase, self).__init__(kwargs)

            self.use_efms = 'efm' in self.measure    

    def get_zs_thetas_dict(self, event, zs, thetas):
        if event is not None:
            zs, thetas = self._measure.evaluate(event)
        elif zs is None or thetas is None:
            raise TypeError('If event is None then zs and thetas cannot also be None')

        return zs, {w: thetas**w for w in self.weight_set}

    def compute_efms(self, event, zs, phats):
        if event is not None:
            zs, phats = self._measure.evaluate(event)
        elif zs is None or phats is None:
            raise TypeError('If event is None then zs and thetas cannot also be None')

        return self.efmset.compute(zs=zs, phats=phats)

    @abstractproperty
    def weight_set(self):
        pass

    @abstractproperty
    def efmset(self):
        pass

    def _batch_compute_func(self, event):
        return self.compute(event, batch_call=True)

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass


###############################################################################
# EFMBase
###############################################################################
class EFMBase(EFBase):

    def __init__(self, kwargs):

        # verify we're using an efm measure
        assert 'efm' in kwargs.setdefault('measure', 'hadrefm'), 'Must use an efm measure.'

        # initialize base class if measure needed
        if not kwargs.pop('no_measure', False):
            super(EFMBase, self).__init__(kwargs)
            self._measure.beta = None

    @abstractmethod
    def compute(self, event=None, zs=None, phats=None):
        if event is not None:
            return self._measure.evaluate(event)
        elif zs is None or phats is None:
            raise ValueError('If event is None then zs and phats cannot be None.')
        return zs, phats

    def _batch_compute_func(self, event):
        return self.compute(event)

    