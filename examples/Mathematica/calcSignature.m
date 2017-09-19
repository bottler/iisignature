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
