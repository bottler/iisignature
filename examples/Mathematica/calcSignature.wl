(* ::Package:: *)

(*If p is a d-dimensional path given as a list of points, where each point is a list of d numbers,
and m is a positive integer,
then Sig[p,m] is the signature of p up to level m excluding the initial 1, given as a list of nested
lists. *)
Sig[p_, m_] := 
 Module[{StraightSig, Chen}, 
  StraightSig[displacement_] := 
   FoldList[Outer[Times, displacement/#2, #1] &, displacement, 
    Range[2, m]]; 
  Chen[d1_, d2_] := 
   Join[{d1[[1]] + d2[[1]]}, 
    Table[d1[[level]] + d2[[level]] + 
      Apply[Plus, 
       MapThread[
        Outer[Times, #1, #2] &, {Take[d1, level - 1], 
         Reverse[Take[d2, level - 1]]}]], {level, 2, m}]];
  Fold[Chen, StraightSig /@ Differences[p]]]

(*LogTensor[Sig[p,m]] is the expanded log signature of p up to level m.*)
LogTensor[t_] := 
 Module[{m, d, terms, powers, ProductTensor}, m = Length[t]; 
  d = Length[t[[1]]];
  ProductTensor[d1_, d2_] := 
   Join[{ConstantArray[0, d]}, 
    Table[Apply[Plus, 
      MapThread[
       Outer[Times, #1, #2] &, {Take[d1, level - 1], 
        Reverse[Take[d2, level - 1]]}]], {level, 2, m}]];
  powers = FoldList[ProductTensor, ConstantArray[t, m]];
  terms = 
   Table[If[OddQ[lev], 1/lev, -1/lev]*powers[[lev]], {lev, 1, m}];
  Apply[Plus, terms]]
  
(*Return the log signature of a path p by projecting the log of 
  the signature onto a basis, as a flat list.
  bases.wl must have been loaded.*)
LogSig[p_, m_, basisGenerator_:GenerateLyndonBasis] :=
  With[{sig=Sig[p,m], d=Length[p[[1]]]},
   LeastSquares[Transpose@LogSigMatrix[d,m,basisGenerator],
                Flatten[LogTensor[sig]]]]

