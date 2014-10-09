============
Installation
============

Dependencies
============

- `Python 2.7`__
.. __: http://www.python.org
- `Scipy >= 0.9.0`__
.. __: http://www.scipy.org
- `Numpy >= 1.6.1`__
.. __: http://numpy.scipy.org
- `Matplotlib >= 1.1.1`__
.. __: http://matplotlib.sourceforge.net
- `PyTables >= 2.1.2 and < 3.0.0`__
.. __: http://www.pytables.org
- `BicycleParameters >= 0.2.0`__
.. __: http://pypi.python.org/pypi/BicycleParameters
- `DynamicistToolKit >= 0.3.5`__
.. __: https://github.com/moorepants/DynamicistToolKit

Installation
============

For ease of setup we recommend setting up a conda_ environment::

   $ conda create -n bdp numpy scipy matplotlib sympy pandas pyyaml "pytables<3.0"
   $ source activate bdp

The remaining dependencies need to be installed with pip::

   (bdp)$ pip install uncertainties "dynamicisttoolkit>=0.3.5"
   (bdp)$ pip install "yeadon>=1.1.1" "BicycleParameters>=0.2.0"

And finally, this package::

   (bdp)$ pip install BicycleDataProcessor

.. _conda: http://conda.pydata.org/
