=============
DataProcessor
=============

Description
===========

This program is setup to process the raw data signals collected from the
instrumented bicycle's data acquisition system (i.e. the output of BicycleDAQ_)

.. _BicycleDAQ: https://github.com/moorepants/BicycleDAQ

License
=======

`BSD 2-Clause License`_, see ``LICENSE.txt``.

.. _BSD 2-Clause License: http://opensource.org/licenses/BSD-2-Clause

Citation
========

If you make use of this data we kindly request that you cite our work, either
[Moore2012]_, the data DOI, and/or other relevant references.

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

The remaining depedencies need to be installed with pip::

   (bdp)$ pip install uncertainties "dynamicisttoolkit>=0.3.5"
   (bdp)$ pip install "yeadon>=1.1.1" "BicycleParameters>=0.2.0"

And finally, this package::

   (bdp)$ pip install BicycleDataProcessor

.. _conda: http://conda.pydata.org/

Usage
=====

Load the prebuilt database file
-------------------------------

The simplest way to get started with the data is to download the latest
database file from::

   $ wget http://files.figshare.com/1710608/instrumented_bicycle_raw_data_h5.tar.bz2
   $ tar -jxvf instrumented_bicycle_raw_data_h5.tar.bz2

And also the bicycle parameter data::

   $ wget http://files.figshare.com/1710525/bicycle_parameters.tar.gz
   $ tar -zxvf bicycle_parameters.tar.gz
   $ rm bicycle_parameters.tar.gz

In your working directory create a ``bdp-defaults.cfg`` and change
``pathToDatabase`` and ``pathToParameters`` to point to the downloaded and
unzipped database file and the ``bicycle-parameters`` data directory,
respectively.

Interact with the data
----------------------

Open a Python command prompt and import the module::

    >>> import bicycledataprocessor as bdp

First load the database::

    >>> dataset = bdp.DataSet()

Now load a run::

    >>> run = bdp.Run('00105', dataset)

Check to make sure the data was properly time synchronized::

    >>> run.verify_time_sync()

The graph that appears shows the mostly downward acceleration signals from the
two accelerometers. These signals are used to synchronize the NI USB-2008 and
the VN-100 data. If these do not match, then the synchronization algorithm
didn't not work and the data may be unusable.

The run has a lot of data associated with it. Firstly, you can print a subset of
the metadata with::

    >>> print(run)

The complete metadata is stored in a dictionary::

    >>> run.metadata

The raw data for each sensor is stored in a dictionary and can be accessed by::

    >>> run.rawSignals

The data for each sensor with calibration scaling can be accessed by::

    >>> run.calibratedSignals

The data for each sensor after truncation based on the time synchronization can
be accessed with::

    >>> run.truncatedSignals

The data for each computed signal is also stored in a dictionary::

    >>> run.computedSignals

The data for each task signal is also stored in a dictionary::

    >>> run.taskSignals

The taskSignals can be plotted::

    >>> run.taskSignals.keys() # see a list of options
    >>> run.plot('SteerAngle', 'RollAngle', 'PullForce')

Export the computed signals as a mat file with::

    >>> run.export('mat')

Build the HDF5 file from raw data
---------------------------------

The second option would be to build the database with the raw data from
BicycleDAQ_. BicycleDAQ_ stores the raw data from trials and calibrations as
Matlab mat files. Then use this module to create the database and fill it with
the data.

The raw trial data can downloaded as so::

   $ wget -O raw-trial-data.zip http://downloads.figshare.com/article/public/1164632
   $ unzip -d raw-trial-data raw-trial-data.zip
   $ rm raw-trial-data.zip

The raw calibration files::

   $ wget -O raw-calibration-data.zip http://downloads.figshare.com/article/public/1164630
   $ unzip -d raw-calibration-data raw-calibration-data.zip
   $ rm raw-calibration-data.zip

And the additional corrupt trial file::

   $ wget -O data-corruption.csv http://files.figshare.com/1696860/data_corruption.csv

Make sure your ``bdp-defaults.cfg`` paths point to the correct directories for
the run mat files (``pathToRunMat``), calibration mat files
(``pathToCalibMat``), the corrupt data file (``data-corruption.csv``).
Optionally the paths can be set as arguments to ``DataSet()``.

Now create an empty database file in the current directory (or to the path
specified in ``bdp-defaults.cfg`` if you've done that).::

   $ python
   >>> import bicycledataprocessor as bdp
   >>> dataset = bdp.DataSet()
   >>> dataset.create_database()

Now, fill the database with the data.::

   >>> dataset.fill_all_tables()

Warnings
========

- The roll angle is not guaranteed to be calibrated in some of the early
  pavillion runs. Caution should be used.
- The first set of pavilion runs with Luke and Charlie are mostly corrupt,
  beware. The corruption column in the ``runTable`` specifies which runs are
  corrupt.
- The yaw angle and lateral deviation values depend on integrating the yaw
  rate. This seems to work for runs that have signals centered around zero, but
  may be wrong for others. (There are plans to fix this for all runs.)

Grant Information
=================

This material is partially based upon work supported by the National Science
Foundation under Grant No. 0928339. Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the authors and do not
necessarily reflect the views of the National Science Foundation.

References
==========

.. [Moore2012] Moore, J. K. Human Control of a Bicycle. University of
   California, Davis. 2012.
