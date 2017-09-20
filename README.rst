The iisignature package
=======================

This package provides tools for calculating the signature and log signature of a data stream. 
These are summary statistics of piecewise linear paths, consisting of iterated integrals, inspired by rough path theory. 
See <http://www2.warwick.ac.uk/jreizenstein/iisignature.pdf> for documentation, the page
<http://www2.warwick.ac.uk/jreizenstein> for more information about the calculations, 
and <https://github.com/bottler/iisignature> for source code.

It is work in progress.

Python
------

Install with::

    pip install iisignature

Don't install this package if you don't have numpy > 1.7 installed. 
On Windows, this package is not usable with Python 2.x. 
For Python 3.5 and 3.6 on Windows, there are precompiled wheels available, you may need to do ``pip install wheel`` to use them.
On other platforms, you will need to be able to compile C++ extensions. 
On a Mac, you will need to have installed Xcode and the Xcode command line tools before doing the installation.
The fastest, on-the-fly compiled versions of the log signature calculations are for Windows, Mac and Linux, on both 32 and 64 bit.

Sometimes there are compile errors on a Mac, if this happens try::

    MACOSX_DEPLOYMENT_TARGET=10.9 pip install iisignature

Use without installing a released version
-----------------------------------------

To install the current "master" development version straight from github you can type::
    pip install git+git://github.com/bottler/iisignature

From a checkout of this repository, you can build the extension and run the tests using::
    python setup.py test

From a checkout of this repository, you can build the extension into this directory itself using::
    python setup.py build_ext --inplace
after which you can use the package in Python in this directory, use the examples in the ``examples`` directory, or add this directory to your PYTHONPATH and then use the package in Python anywhere.


Authors
-------

* Dr Benjamin Graham
* Jeremy Reizenstein

Thanks
------

This package includes BCH coefficients from Fernando Casas and Ander Murua.
<http://arxiv.org/abs/0810.2656>
