gcc directory
=============

This directory contains a few scripts which build (something equivalent to) the addin
or code using its functionality using clang or GCC (for me, on Ubuntu) without touching
any of the Python setuptools/distutils stuff.. Nothing here affects the iisignature addin
itself. These scripts should not produce any files outside this directory,
and nothing outside this directory refers to anything inside this directory.

* buildAddin.sh: builds the addin

* standalone.cpp: a c++ program using functionality from the header files of iisignature.

* buildStandalone.sh: builds it

* vg.sh: runs valgrind on it

* view.sh: views the output in kcachegrind


Memory profiling
----------------

The standalone program defined in mem_profile.cpp calculates a single log signature.
It is built with `mem_profile_build.sh`. `mem_profile_run.py` uses the `massif` tool
from `valgrind` to work out the total memory usage. `mem_profile_analyse.py` plots
results.