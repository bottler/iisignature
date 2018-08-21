iisignature examples
====================

The files in this folder beginning with "iisignature" contain useful
python functionality which builds on the functionality in iisignature.
They are not part of the iisignature package build.
They can have dependencies on other python packages which iisignature
does not depend on - e.g. Theano and tensorflow.

* iisignature_theano.py and iisignature_tensorflow.py provide theano and tensorflow operations
  which mirror sig, sigjoin and sigscale from iisignature. demo_keras.py shows how
  they might be used very simply in a Keras layer.

* iisignature_recurrent_keras.py shows a recurrent network keras layer which uses signatures of the history of
  its units

* iisignature_torch.py provides pytorch operations
  which mirror sig, sigjoin and sigscale from iisignature. demo_torch.py shows how they can be called very simply.

* iisignature_recurrent_torch.py shows a recurrent network pytorch layer which uses signatures of the history of
  its units

* iisignature_match_esig.py wraps essential functionality from iisignature in an interface
  which matches esig.tosig (formerly called sigtools)
  from CoRoPa as closely as possible.

The files beginning with "demo" are just demonstrations of this functionality.

If you have not installed iisignature but have the git checkout of the
code, then you should be able to run the demo files once you have run
``python setup.py build_ext --inplace`` in the root of the tree,
because they append ``..`` to the ``PYTHONPATH``.

-----

Many of the other files are freestanding programs.

* figure8Fibonacci.py shows how to build paths which traverse a figure of 8 multiple times whose signature vanishes below level 5, 8, or 13.

* freehand_draw.py prints out log signatures of paths you draw freehand with the mouse.

* innerProductLogSig.py shows how to find the L2 inner product of the tensor-space-expanded versions of log signatures when they are not tensor-space-expanded.

* leak_check.py runs functions from iisignature many times - I can watch memory usage while this happens to try to diagnose memory leaks.

* make_axis_path.py is a simple utility to make a path which moves single units along the axes matching a given sequence.
  
* protectedRotinvExperiment.py shows how you might, on Windows and Linux, experiment with high levels of some calculation (rotational invariants in this case) without fear you will stall your computer by eating all your RAM.

The **Mathematica** folder contains some signature related Mathematica stuff.

The **matlab** folder contains some signature related matlab stuff.

The **R** folder contains some signature related R stuff.
