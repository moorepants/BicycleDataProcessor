============
Installation
============

Dependencies
============

- `Python 2.7`_
- `NumPy >= 1.6.1`_
- `SciPy >= 0.9.0`_
- `Matplotlib >= 1.1.1`_
- `PyTables >= 2.1.2 and < 3.0.0`_
- `BicycleParameters >= 0.2.0`_
- `DynamicistToolKit >= 0.3.4`_

.. _Python 2.7: http://www.python.org
.. _NumPy >= 1.6.1: http://numpy.scipy.org
.. _SciPy >= 0.9.0: http://www.scipy.org
.. _Matplotlib >= 1.1.1: http://matplotlib.sourceforge.net
.. _PyTables >= 2.1.2 and < 3.0.0: http://www.pytables.org
.. _BicycleParameters >= 0.2.0: http://pypi.python.org/pypi/BicycleParameters
.. _DynamicistToolKit >= 0.3.4: https://pypi.python.org/pypi/DynamicistToolKit

Installation
============

For ease of setup we recommend setting up a conda_ environment::

   $ conda create -n bdp numpy scipy matplotlib "pytables<3.0" pyyaml
   $ source activate bdp

The remaining dependencies need to be installed with pip::

   (bdp)$ pip install "uncertainties>2.0.0" "DynamicistToolKit>=0.3.4"
   (bdp)$ pip install "yeadon>=1.1.1" "BicycleParameters>=0.2.0"

And finally, this package::

   (bdp)$ pip install BicycleDataProcessor

.. _conda: http://conda.pydata.org/
