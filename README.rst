The iisignature package
=======================

This package provides tools for calculating the signature and log signature of a data stream. These are summary statistics of piecewise linear paths, consisting of iterated integrals, inspired by rough path theory. See <http://www2.warwick.ac.uk/jreizenstein> for documentation and more information about the calculations, and <https://github.com/bottler/iisignature> for source code.

It is work in progress.

Python
------

Install with::

    pip install iisignature

For the moment, don't install this package if you don't have numpy > 1.7 installed. On Windows, this package is not usable with Python 2.x. For Python 3.5 and 3.6 on Windows, there are precompiled wheels available, you may need to do ``pip install wheel`` to use them. On other platforms, you will need to be able to compile C++ extensions. The fastest, on-the-fly compiled versions of the log signature calculations are for Windows (32 and 64 bit) and 64 bit linux.

Authors
-------

* Dr Benjamin Graham
* Jeremy Reizenstein

Thanks
------

This package includes BCH coefficients from Fernando Casas and Ander Murua.
<http://arxiv.org/abs/0810.2656>
