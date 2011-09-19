=============
DataProcessor
=============

Description
===========
This program is setup to process the raw data signals collected from the
instrumented bicycle's data aquistion system (i.e. the output of BicycleDAQ_)

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
- `DynamicistToolKit 0.1.0dev`__
.. __: https://github.com/moorepants/DynamicistToolKit
- `BicycleParameters 0.1.3`__
.. __: http://pypi.python.org/pypi/BicycleParameters

Usage
=====

Build the PyTables HDF5 File
----------------------------

The simplest way to get started with the data is to download the latest
database file. Once it is uncompressed, the file is ready for use.

The second option would be to build the database with the raw data from
BicycleDAQ_. BicycleDAQ_ stores the raw data as matlab matfiles. These need to be
converted to equivalent HDF5 files to be able to load into the master database
file. Use the m-file `tools/fill_h5.m` to convert the run and calibration data
into HDF5 files.

>>> import BicycleDataProcessor as bdp

First create an empty database file in the current dirctory.

>>> bdp.create_database()

Now, fill the database with the data.

>>> bdp.fill_tables()
