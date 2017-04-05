VS directory
============

This directory contains a Visual Studio 2017 Solution `try\try.sln`
which contains a few example projects. Nothing here affects the 
iisignature addin itself. Building this solution should not produce
any files outside this directory, and nothing outside this directory
refers to anything inside this directory. The output directory of all 
these projects is the try project's one.

The buildso project
-------------------

This builds an addin just like the iisignature addin, except not using 
setuptools. Unlike the rest of this solution, this uses the 2015 toolset
to match python 3.5 and 3.6. You will need to change the project settings so
that they point to your installation(s) of python. On release mode this 
means you have a normal visual studio experience of the addin.

In order for the prepare() function to work, you need to copy the iisignature_data
directory from the root of this tree to the relevant output directory
(i.e. the one for the try project). Or, I guess, anywhere on your PYTHONPATH.

The debug mode build of this project is only of limited use. It lets you debug 
part of the functionality of iisignature. It will require you to 
have checked the right options when you installed
python, so that you have debugging files for python. Your debug executable
will be python_d.exe in the python install directory.
I haven't bothered playing with debug builds of numpy. The debug version of this 
project is therefore a build of iisignature which doesn't depend on numpy at all. 
Several functions in the addin are not available, and part of the prepare function
is omitted. The macro IISIGNATURE_NO_NUMPY is defined (and that is the purpose
of this macro).


The try project
---------------

This is a simple standalone application which uses the functionality of iisignature.
Python is not referenced. If you want to do calculations in C++ you can use
this project. The particular code is just me playing.


the arbprec project
-------------------

This is a standalone application which depends on boost in a header only fashion to compare 
some signature calculations with arbitrary precision versions of themselves.

