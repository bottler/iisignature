from setuptools import setup, Extension
import numpy

xtn = Extension('iisignature', ['src/baresig.cpp'], extra_compile_args=['-std=c++11'], include_dirs=[numpy.get_include()])


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='iisignature',
      version='0.1',
      ext_modules=[xtn],
      description='Iterated integral signature calculations',
      long_description=readme(),
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 1 - Planning',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Financial and Insurance Industry',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Mathematics',
          
          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: MIT License',
          
          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          ],
      url='http://www2.warwick.ac.uk/fac/cross_fac/complexity/people/students/dtc/students2013/reizenstein/',
      author='Jeremy Reizenstein',
      author_email='j.f.reizenstein@warwick.ac.uk',
      keywords = ["signature", "logsignature"],
      license='MIT',
      packages=[],
      zip_safe=False,           
      install_requires=['numpy>1.7'], #For now, this is redundant as we've died above here on an import statement
)
#should add source url from github like at https://www.python.org/dev/peps/pep-0426/
