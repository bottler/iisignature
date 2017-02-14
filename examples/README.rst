The files in this folder beginning with "iisignature" contain useful
python functionality which builds on the functionality in iisignature.
They are not part of the iisignature package build.
They can have dependencies on other python packages which iisignature
does not depend on - e.g. Theano.

The files beginning with "demo" are just demonstrations of this functionality.

If you have not installed iisignature but have the git checkout of the
code, then you should be able to run the demo files once you have run
`python setup.py build_ext --inplace` in the root of the tree,
because they append `..` to the `PYTHONPATH`.
