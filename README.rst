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
likely work with other versions.

- `Python 2.7.1`__
.. __: http://www.python.org
- `Scipy 0.9.0`__
.. __: http://www.scipy.org
- `Numpy 1.5.1`__
.. __: http://numpy.scipy.org
- `Matplotlib 0.99.3`__
.. __: http://matplotlib.sourceforge.net
- `PyTables 2.1.2`__
.. __: http://www.pytables.org
- `BicycleParameters 0.1.3`__
.. __: http://pypi.python.org/pypi/BicycleParameters
- `DynamicistToolKit 0.1.0dev`__
.. __: https://github.com/moorepants/DynamicistToolKit

Usage
=====

Load prebuilt database file
---------------------------

The simplest way to get started with the data is to download the latest
database file from:

http://mae.ucdavis.edu/~biosport/InstrumentedBicycleData/InstrumentedBicycleData.h5.bz2

Uncompress the file into your `BicycleDataProcessor` directory, the file is
ready for use. Open a python command prompt and import the module::

    >>> import bicycledataprocessor.main as bdp

First load the database as read-only::

    >>> database = bdp.load_database()

Now load a run::

    >>> run = bdp.Run('00105', database, <pathToParameterData>, filterSigs=True)

The `<pathToParameterData>` needs to point to the data directory associated
with the BicycleParameters module and should contain Jason and the Rigd
bicycle. The `filterSigs` will apply a filter to the signals to remove some of
the noise, it is optional.

Check to make sure the data was properly time synchronized::

    >>> run.plot('AccelerationZ', '-AccelerometerAccelerationY', signalType='truncated')

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

The computedSignals can be plotted::

    >>> run.computedSignals.keys() # see a list of options
    >>> run.plot('SteerAngle', 'RollAngle', 'PullForce')

Export the computed signals as a mat file with::

    >>> run.export('mat')

Build the PyTables HDF5 File from raw data
------------------------------------------

The second option would be to build the database with the raw data from
BicycleDAQ_. BicycleDAQ_ stores the raw data as matlab matfiles. These need to be
converted to equivalent HDF5 files to be able to load into the master database
file. Use the m-file `tools/fill_h5.m` to convert the runs and calibration data
into HDF5 files. Then use this module to create the database and fill it with
the data. First create an empty database file in the current directory.::

    >>> bdp.create_database()

Now, fill the database with the data.::

    >>> bdp.fill_tables()

Warnings
========
- The roll angle is not guaranteed to be calibrated in some of the early
  pavillion runs. Check this.
- The system currently only loads Jason onto the bicycle. This shouldn't affect
  anything major on runs with Charlie and Luke, but there are some small
  discrepancies.
- The current pavilion runs with Luke and Charlie are mostly corrupt, be ware.
- The yaw angle and lateral deviation values depend on integrating the yaw
  rate. This seems to work for runs that have signals centered around zero, but
  are definitely wrong for others. (There are plans to fix this for all runs.)
