(* ::Package:: *)

SetDirectory[NotebookDirectory[]]
<< "bch.m"
view=Manipulate[(range3 = 0.4; range4 = 0.2;(*regular=0.051;*)f = False; 
  sig = ConstantArray[0, 8];
  If[solve,
   (sig = {level1 [[1]], level1 [[2]], level2, level3[[1]], 
      level3[[2]], level4a, level4b, level4c};
    min = 
     NMinimize[
      Total[(bch[{a1, a2}, {b1, b2}, {c1, c2}, {d1, d2}] - sig)^2] + 
       regular * Total[{a1, a2, b1, b2, c1, c2, d1, d2}^2], {a1, a2, 
       b1, b2, c1, c2, d1, d2} , Method -> method];
    regularizedpart = 
     Total[{a1, a2, b1, b2, c1, c2, d1, d2}^2] /. Last[min];
    a = {a1, a2} /. Last[min]; b = {b1, b2} /. Last[min]; 
    c = {c1, c2} /. Last[min]; d = {d1, d2} /. Last[min];
    ab = a + b; abc = ab + c; abcd = abc + d;
    rep1 = First[min] - regular*regularizedpart; 
    rep2 = regularizedpart; rep3 = regular*regularizedpart),
   (b = ab - a; c = abc - ab; d = abcd - abc; rep1 = 0; rep2 = 0; 
    rep3 = 0; mysig = bch[a, b, c, d]; 
    level1 = {mysig[[1]], mysig[[2]]};
    level2 = mysig[[3]]; level3 = {mysig[[4]], mysig[[5]]}; 
    level4a = mysig[[6]]; level4b = mysig[[7]]; level4c = mysig[[8]])];
  Column[{ListPlot[{{0, 0}, a, ab, abc, abcd}, Joined -> True, Mesh -> All, 
    ImageSize -> Large, 
    PlotRange -> {{-scale, scale}, {-scale, scale}}, 
    AspectRatio -> 1],{rep1, rep2, rep3,
   Framed[MatrixForm[Transpose[{bch[a, b, c, d], sig}]]]}}])
 , {solve, {True, False}}, {{level1, {1, 1}}, {-2, -2}, {2, 2}, 
  Appearance -> "Labeled"}, {{level2, 1.1, "12"}, -2, 2, 
  Appearance -> 
   "Labeled"}, {{level3, {0.1, 0.1}}, {-range3, -range3}, {range3, 
   range3}, Appearance -> "Labeled"}, {{level4a, 0, "1112"}, -range4, 
  range4, Appearance -> "Labeled"}, {{level4b, 0, "1122"}, -range4, 
  range4, Appearance -> "Labeled"}, {{level4c, 0, "1222"}, -range4, 
  range4, Appearance -> "Labeled"}, {{scale, 5}, {2, 5, 10, 20, 
   40}}, {{regular, 0.2}, 0, 5, Appearance -> "Labeled"},
 {a, Locator}, {ab, Locator}, {abc, Locator}, {abcd, 
  Locator}, {method, {NelderMead, DifferentialEvolution, RandomSearch,
    SimulatedAnnealing, {SimulatedAnnealing, PerturbationScale -> 3, 
    SearchPoints -> 100}} , Appearance -> "Row"}, 
 ControlPlacement -> Right, ContinuousAction -> False]




