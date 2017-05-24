** iisignature examples

The files in this folder beginning with "iisignature" contain useful
python functionality which builds on the functionality in iisignature.
They are not part of the iisignature package build.
They can have dependencies on other python packages which iisignature
does not depend on - e.g. Theano.

The files beginning with "demo" are just demonstrations of this functionality.

If you have not installed iisignature but have the git checkout of the
code, then you should be able to run the demo files once you have run
``python setup.py build_ext --inplace`` in the root of the tree,
because they append ``..`` to the ``PYTHONPATH``.

leak_check.py runs functions from iisignature many times - I can watch 
memory usage while this happens to try to diagnose memory leaks.

innerProductLogSig.py shows how to find the L2 inner product of the tensor-space-expanded 
versions of log signatures when they are not tensor-space-expanded.

The *Mathematica* folder contains some signature related Mathematica stuff.

