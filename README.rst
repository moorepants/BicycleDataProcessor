=============
DataProcessor
=============

Description
===========

This program is setup to process the raw data signals collected from the
instrumented bicycle's data acquisition system (i.e. the output of BicycleDAQ_)

.. _BicycleDAQ: https://github.com/moorepants/BicycleDAQ

Dependencies
============

These are the versions that I tested the code with, but the code will most
likely work with newer versions.

- `Python 2.7.1`__
.. __: http://www.python.org
- `Scipy 0.10.0`__
.. __: http://www.scipy.org
- `Numpy 1.5.1`__
.. __: http://numpy.scipy.org
- `Matplotlib 0.99.3`__
.. __: http://matplotlib.sourceforge.net
- `PyTables 2.1.2`__
.. __: http://www.pytables.org
- `BicycleParameters`__ (either checkout the ``dissertation`` tag or use the
  latest master)
.. __: http://pypi.python.org/pypi/BicycleParameters
- `DynamicistToolKit`__ (either checkout the ``dissertation`` tag or use the
  latest master)
.. __: https://github.com/moorepants/DynamicistToolKit

Usage
=====

Load prebuilt database file
---------------------------

The simplest way to get started with the data is to download the latest
database file from::

   $ wget http://mae.ucdavis.edu/~biosport/InstrumentedBicycleData/InstrumentedBicycleData.h5.bz2

Uncompress the file and it is ready for use.::

   $ bzip2 -d InstrumentedBicycleData.h5.bz2

Now edit ``defaults.cfg`` and change ``pathToDatabase`` and
``pathToParameters`` to point to the downloaded and unzipped database file and
the ``BicycleParameters`` data folder, respectively.

Open a python command prompt and import the module::

    >>> import bicycledataprocessor as bdp

First load the database as read-only::

    >>> dataset = bdp.DataSet()

Now load a run::

    >>> run = bdp.Run('00105', dataset)

The `filterSigs` will apply a filter to the signals to remove some of the
noise, it is optional.

Check to make sure the data was properly time synchronized::

    >>> run.verify_time_sync()

The graph that appears shows the mostly downward acceleration signals from the
two accelerometers. These signals are used to synchronize the NI USB-2008 and
the VN-100 data. If these do not match, then the synchronization algorithm
didn't not work and the data may be unusable.

The run has a lot of data associated with it. Firstly, you can print a subset of
the metadata with::

    >>> print run

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

Build the PyTables HDF5 File from raw data
------------------------------------------

The second option would be to build the database with the raw data from
BicycleDAQ_. BicycleDAQ_ stores the raw data as Matlab mat files. Then use this
module to create the database and fill it with the data.

Make sure your ``defaults.cfg`` paths point to the correct directories for the
run mat files (``pathToRunMat``), calibration mat files (``pathToCalibMat``),
the corrupt data file (``data-corruption.csv``).

First create an empty database file in the current directory.::

    >>> import bicycledataprocessor as bdp
    >>> dataset = bdp.DataSet()
    >>> dataset.create_database()

Now, fill the database with the data.::

    >>> dataset.fill_all_tables()

The path to all of the raw data must be specififed in the ``defaults.cfg`` or
as arguments to ``DataSet()``.

Warnings
========

- The roll angle is not guaranteed to be calibrated in some of the early
  pavillion runs. Check this.
- The first set of pavilion runs with Luke and Charlie are mostly corrupt,
  beware. The corruption column in the runTable specifies which runs are
  corrupt.
- The yaw angle and lateral deviation values depend on integrating the yaw
  rate. This seems to work for runs that have signals centered around zero, but
  are definitely wrong for others. (There are plans to fix this for all runs.)
