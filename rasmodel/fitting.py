r"""
A cookbook wrapper around the parameter optimization algorithm in
scipy.optimize. Drawn from
http://www.scipy.org/Cookbook/FittingData#head-5eba0779a34c07f5a596bbcf99dbc7886eac18e5.

This code is very useful for performing basic fitting of various mathematical
functions to data.

Example usage:

.. ipython:: python

    from rasmodel import fitting
    import numpy as np
    t = np.linspace(0, 100, 50)
    data = 3*t + 4
    m = fitting.Parameter(2)
    b = fitting.Parameter(5)
    def fit_func(x): return (m()*x + b())
    fitting.fit(fit_func, [m, b], data, t)
    print "m: %f\nb: %f" % (m(), b())
"""

import numpy as np
from scipy import optimize
from pysb.integrate import Solver
from matplotlib import pyplot as plt

class Parameter:
    """A simple object wrapper around parameter values.

    The parameter value is accessed using the overloaded call method, e.g.
    ``k()``.

    Parameters
    ----------
    value : number
        The initial guess for the parameter value.
    """
    def __init__(self, value):
            self.value = value

    def set(self, value):
        """Set the value."""
        self.value = value

    def __call__(self):
        """Get the value by calling the parameter, e.g. k()."""
        return self.value

def fit(function, parameters, y, x=None, log_transform=True, maxfev=100000):
    """Fit the function to the data using the given parameters.

    Creates a wrapper around the fitting function and passes it to
    `scipy.optimize.leastsq`. Every evaluation of the wrapper function assigns
    new values (the values being tested at that evaluation) to the parameter
    objects in ``parameters`` as a side effect. This way, once
    `scipy.optimize.leastsq` is finished executing, the parameters have their
    new, optimized values.

    Parameters
    ----------
    function : function
        The fit function should itself be a closure, referencing the
        parameter values after they have been declared in the calling
        scope. The function should be able to operate on a input
        matching the argument ``x``, and produce an output that can
        be compared to the argument ``y``.
    parameters : list of :py:class:`Parameter`
        The parameters used in the fitting function.
    y : numpy.array
        The output of ``function`` evaluated on ``x`` will be compared to
        ``y``.
    x : numpy.array
        The input to ``function``, e.g., a time vector. If ``None`` (the
        default), an ordinal vector running from [0 ... y.shape[0]] is used.
    maxfev : int
        Maximum function evaluations. Passed to ``scipy.optimize.leastsq``
        as a keyword argument.

    Returns
    -------
    tuple: (residuals, result)
        The first entry in the tuple is a num_timepoints length array
        containing the residuals at the best fit parameter values. The second
        element is the result object returned by scipy.optimize.leastsq.
    """
    def f(params):
        i = 0
        for p in parameters:
            if log_transform:
                p.set(10 ** params[i])
            else:
                p.set(params[i])
            i += 1
        err = y - function(x)
        err = err[~np.isnan(err)]
        return err

    if x is None: x = np.arange(y.shape[0])

    if log_transform:
        p = [np.log10(param()) for param in parameters]
    else:
        p = [param() for param in parameters]

    result = optimize.leastsq(f, p, ftol=1e-12, xtol=1e-12, maxfev=maxfev,
                              full_output=True)
    # At this point the parameter instances should have their final fitted
    # values, NOT log transformed. To get the residuals at this final value,
    # we need to pass in log transformed values
    if log_transform:
        residuals = f([np.log10(param()) for param in parameters])
    else:
        residuals = f([param() for param in parameters])

    return (residuals, result)
