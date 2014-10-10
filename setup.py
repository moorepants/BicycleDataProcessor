#!/usr/bin/env python

from setuptools import setup, find_packages

exec(open('bicycledataprocessor/version.py').read())

setup(
    name='BicycleDataProcessor',
    version=__version__,
    author='Jason Keith Moore',
    author_email='moorepants@gmail.com',
    packages=find_packages(),
    url='http://github.com/moorepants/BicycleDataProcessor',
    license='LICENSE.txt',
    description='Processes the data collected from the instrumented bicycle.',
    long_description=open('README.rst').read(),
    install_requires=['numpy>=1.6.1',
                      'scipy>=0.9.0',
                      'matplotlib>=1.1.1rc',
                      'tables<3.0',
                      'DynamicistToolKit>=0.3.4',
                      'BicycleParameters>=0.2.0'],
    extras_require={'doc': ['sphinx', 'numpydoc']},
    tests_require=['nose'],
    test_suite='nose.collector',
    classifiers=['Programming Language :: Python',
                 'Programming Language :: Python :: 2.7',
                 'Operating System :: OS Independent',
                 'Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Natural Language :: English',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Physics']
)
