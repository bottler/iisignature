Some mathematica stuff here for playing with signatures. Not related to iisignature.

* bch.m: generated code to calculate the lyndon-basis logsignature up to level 4 of a path in 2d given as 4 displacements. This was produced by logsignature.py.

* view.m: a large Manipulate widget for demonstrating the log signature up to level 4 (shown in widgets) of a path given by 4 straight lines (uses bch.m). Used with care, it can be instructive to (with solve unticked) see how moving the path a bit changes the signature a bit, and (with solve ticked) see how moving the logsignature a bit might move the path.


* calcSignature.wl: calculate the signature of a path numerically, like iisignature.sig .

* parametricCurves.wl: signatures of parametric curves.

* bases.wl: Some free Lie algebra basis calculations.

* circleExample.wl: The signature of the unit circle up to level 6, for reference.
