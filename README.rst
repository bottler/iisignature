The iisignature package
=======================

This package provides tools for calculating the signature and log signature of a data stream. These are summary statistics of piecewise linear paths, consisting of iterated integrals, inspired by rough path theory. See <http://www2.warwick.ac.uk/jreizenstein> for more information about the calculations.

It is work in progress.

Python
------

Install with::

    pip install iisignature

For the moment, don't install this package if you don't have numpy > 1.7 installed, or if you can't compile C++ extensions. Log signature calculations (except the slow version) are designed for x86-64 and have only been tested on Linux. They will probably crash on other platforms.

Authors
-------

* Dr Benjamin Graham
* Jeremy Reizenstein

Thanks
------

This package includes BCH coefficients from Fernando Casas and Ander Murua.
<http://arxiv.org/abs/0810.2656>
