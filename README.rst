The sigstreamlie package
========================

This package provides tools for calculating the signature and log signature of a data stream. These are summary statistics of piecewise linear paths, consisting of iterated integrals, inspired by rough path theory. See <http://www2.warwick.ac.uk/fac/cross_fac/complexity/people/students/dtc/students2013/reizenstein/> for more information about the calculations.

For the moment, don't install this package if you don't have numpy > 1.7.1 installed, or if you can't compile C++ extensions. Log signature calculation will probably crash if not running on an x86-64 Linux box.
