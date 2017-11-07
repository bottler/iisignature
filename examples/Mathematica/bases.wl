(* ::Package:: *)

(*Tools for doing log signature / free Lie algebra basis calculations.
These functions are written for brevity and partly exposition, not for
speed.
*)

LengthOfExpression[a_Integer]:=1;
LengthOfExpression[{a_,b_}]:=LengthOfExpression[a]+LengthOfExpression[b];

(*make a Hall set of bracketed expressions 
according to the sort given by less, which must satisfy
less[{a,b},b] for all a,b
E.g. Reutenauer page 84*)
GenerateHallBasis[d_Integer, 1, less_]:={Range[d]}
GenerateHallBasis[d_Integer, m_Integer, less_]:=
 With[{known = GenerateHallBasis[d,m - 1,less]}, 
  With[{new = 
     Table[Join @@ 
       Table[If[And[less[x,y],
         Or[IntegerQ[x], Not[less[x[[2]], y]]]], {x, y}, 
         Nothing], {x, known[[firstLev]]}, {y, 
         known[[m - firstLev]]}], {firstLev, 1, m-1}]},
    Append[known,Join@@new]]];

(*reverse all the brackets in a bracketed expression*)
ReverseAllBrackets[a_Integer]:=a;
ReverseAllBrackets[{a_,b_}]:={ReverseAllBrackets[b],ReverseAllBrackets[a]};

LessExpressionStandardHall[a_, b_] := 
 With[{ll = LengthOfExpression[a], lr = LengthOfExpression[b]}, 
  If[ll == lr, 
   If[ll == 1, b < a, 
    If[LessExpressionStandardHall[a[[1]], b[[1]]], True, 
     If[LessExpressionStandardHall[b[[1]], a[[1]]], False, 
      LessExpressionStandardHall[a[[2]], b[[2]]]]]], lr < ll]];

(*lexicographic on the foliage. Obvs this will go haywire if d exceeds 9.*)
LessExpressionLyndon[a_,b_]:=
 1==Order[StringJoin@@TextString/@Flatten@{a},StringJoin@@TextString/@Flatten@{b}];

GenerateLyndonBasis[d_Integer,m_Integer] := GenerateHallBasis[d,m,LessExpressionLyndon];

(*Generate the same basis used in Coropa/ESig*)
GenerateStandardHallBasis[d_Integer,m_Integer] := 
  Map[ReverseAllBrackets, GenerateHallBasis[d,m,LessExpressionStandardHall],{2}];
      

(*ExpandBracketedExp[expression,d] returns the expanded value of expression
in Tensor space, only for the relevant level (i.e. LengthOfExpression[expression]). 
The "Flat" version Flattens the level. 
*)
ExpandBracketedExpFlat[x_Integer,d_Integer] := 
 Join[ConstantArray[0, x - 1], {1}, ConstantArray[0, d - x]]; 
ExpandBracketedExpFlat[{x_,y_},d_Integer] :=
 With[{a = ExpandBracketedExpFlat[x,d], b = ExpandBracketedExpFlat[y,d]}, 
  Flatten[KroneckerProduct[a, b]] - Flatten[KroneckerProduct[b, a]]];
  
ExpandBracketedExp[x_Integer,d_Integer] := 
 Join[ConstantArray[0, x - 1], {1}, ConstantArray[0, d - x]]; 
ExpandBracketedExp[{x_,y_},d_Integer] :=
 With[{a = ExpandBracketedExp[x,d], b = ExpandBracketedExp[y,d]}, 
  TensorProduct[a, b] - TensorProduct[b, a]];

(*Like LogTensor*)
ExponentiateTensor[t_] := 
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
   Table[( 1/Factorial[lev])*powers[[lev]], {lev, 1, m}];
  Apply[Plus, terms]];

(*These "Arbitrary" functions generate expressions like those in the logsigdemo document*)
ArbitraryLogSig[d_Integer,m_Integer]:=With[{bas=GenerateLyndonBasis[d,m]},
  Table[Plus@@Table[ExpandBracketedExp[xx,d] Unique[],{xx,x}],{x,bas}]];

ArbitraryLogSigAndSig[d_Integer,m_Integer]:=
  With[{l=ArbitraryLogSig[d,m]},{l,ExponentiateTensor[l]}];

(*Outputs the signature with level m multiplied by (m!), don't forget!*)
ArbitraryLogSigAndSigNoDenominators[d_Integer,m_Integer]:=
	With[{l=ArbitraryLogSig[d,m]},With[{e=ExponentiateTensor[l]},
	    {l,Table[Expand[Factorial[x]e[[x]]],{x,1,m}]}]];


(*Generate the matrix btw the basis and tensor space for each level separately*)
LogSigMatrices[d_,m_,basisGenerator_:GenerateLyndonBasis]:=
    With[{bas=basisGenerator[d,m]},
	  Table[Table[ExpandBracketedExpFlat[xx,d],{xx,x}],{x,bas}]];
	  
(*Generate the matrix btw the basis and tensor space - block diagonal of LogSigMatrices.*)
LogSigMatrix[d_,m_,basisGenerator_:GenerateLyndonBasis]:=
	Fold[ArrayFlatten[{{#, 0}, {0, #2}}] &, LogSigMatrices[d,m,basisGenerator]];



