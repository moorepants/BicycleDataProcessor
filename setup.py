from distutils.core import setup

setup(
      name='BicycleDataProcessor',
      version='0.1.0dev',
      author='Jason Keith Moore',
      author_email='moorepants@gmail.com',
      packages=['bicycledataprocessor',
                'bicycledataprocessor.test'],
      url='http://github.com/moorepants/BicycleDataProcessor',
      license='LICENSE.txt',
      description='''Processes the data collected from the instrumented
bicycle.''',
      long_description=open('README.rst').read()
)
