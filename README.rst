The iisignature package
=======================

This package provides tools for calculating the signature and log signature of a data stream.
These are summary statistics of piecewise linear paths, consisting of iterated integrals, inspired by rough path theory.
See <https://github.com/bottler/phd-docs/blob/main/iisignature.pdf> for documentation, the page
<https://github.com/bottler/phd-docs> for more information about the calculations,
and <https://github.com/bottler/iisignature> for source code.

It is work in progress.

Python
------

Install a released version, which may be missing recent bugfixes, with::

    pip install iisignature

Don't install this package if you don't have numpy > 1.7 installed.
For Python 3.5, 3.6, 3.7 and 3.8 on Windows, there are precompiled wheels available, you may need to do ``pip install wheel`` to use them.
On other platforms, you will need to be able to compile C++ extensions.
On a Mac, you will need to have installed Xcode and the Xcode command line tools before doing the installation.
The fastest, on-the-fly compiled versions of the log signature calculations are for Windows, Mac and Linux, on both 32 and 64 bit.

Sometimes there are compile errors on a Mac, if this happens try::

    MACOSX_DEPLOYMENT_TARGET=10.9 pip install iisignature

Use without installing a released version (START HERE)
------------------------------------------------------

To install the current "main" development version straight from github you can type::

    pip install git+https://github.com/bottler/iisignature

From a checkout of this repository, you can build the extension and run the tests using::

    python setup.py test

From a checkout of this repository, you can build the extension into this directory itself using::

    python setup.py build_ext --inplace

after which you can use the package in Python in this directory, use the examples in the ``examples`` directory, or add this directory to your PYTHONPATH and then use the package in Python anywhere.


Paper
-----

A paper in TOMS <https://dl.acm.org/doi/10.1145/3371237> accompanies the library.
It can also be found on arxiv at <https://arxiv.org/abs/1802.08252>.
If you find iisignature useful in your research then please cite:

    @article{iisignature,
      title={Algorithm 1004: The iisignature Library:
              Efficient Calculation of Iterated-Integral Signatures and Log Signatures},
      author={Jeremy Reizenstein and Benjamin Graham},
      journal={ACM Transactions on Mathematical Software (TOMS)},
      year={2020}
    }

Authors
-------

* Benjamin Graham
* Jeremy Reizenstein

Thanks
------

This package includes BCH coefficients from Fernando Casas and Ander Murua.
<http://arxiv.org/abs/0810.2656>
