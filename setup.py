from distutils.core import setup

setup(
      name='DataProcessor',
      version='0.1.0dev',
      author='Jason Keith Moore',
      author_email='moorepants@gmail.com',
      packages=['dataprocessor',
                'dataprocessor.test'],
      url='http://github.com/moorepants/InstrumentedBicycle/DataProcessor',
      license='LICENSE.txt',
      description='Processes the data collected from the instrumented bicycle.'
      long_description=open('README.rst').read(),
     )
