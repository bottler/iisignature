(* ::Package:: *)

(*Functions for calculating the signature of a parametric curve.
There are two versions of everything:
* An "Exps" is a curve specified as a list of expressions and a symbol.
  e.g. {{Cos[x],Sin[x],x},x}.
* A "Fns" is a curve specified as a list of functions.
  e.g. {Cos, Sin, Identity}.
We consider them on the interval [0,max].
You can calculate the signature of a curve, and also join, shift, and evaluate curves,
and add a time dimension to them.
You can plot them if they are 2 or 3 dimensional.*)

(** 1. Exps versions. *)

(*Plot uses most of the rainbow red\[Rule]blue. Note that uu ranges over [0,1], it isn't the parameter.*)
PlotExps[exps_,sym_Symbol,max_]:=
  If[Length[exps] == 2, ParametricPlot, ParametricPlot3D][exps,{sym,0,max},ColorFunction->Function[{x,y,uu},Hue[0.75uu]]]
ShiftExps[exps_,sym_Symbol,by_]:=Table[ff/.sym->sym+by,{ff,exps}]
EvaluateExps[exps_,sym_Symbol,paramValue_]:=exps/.sym->paramValue
JoinExps[{f_,sym1_Symbol},{g_,sym2_Symbol},break_]:=
   Table[With[{diff=(tt[[1]]/.sym1->break)-(tt[[2]]/.sym2->0)},
        Piecewise[{{tt[[1]],sym1<break}},diff+(tt[[2]]/.sym2->sym1-break)]],
   {tt,Transpose[{f,g}]}]
AddTimeToExp[exp_,sym_Symbol]:=Append[exp,sym]
SwapExps[{a_,b_}]:={b,a}
ReflectExpsThroughOrigin[a_]:=-a
RotateExpsAnticlockwise[{a_,b_},rad_:(Pi/2)]:={a Cos[rad]-b Sin[rad],a Sin[rad]+b Cos[rad]}

(*SigExpsFull gives the signature including the initial 1 and an extra level of nesting.*)
(*The Assuming often seems like a required thing*)
(*SigExpsFull[exps_, sym_Symbol, u_, m_] := 
  NestList[Outer[
     Assuming[sym>0,Integrate[# D[#2,sym]/.sym\[Rule]t, {t, 0, sym}]] & , #, 
     exps] & , {1}, m]/.sym\[Rule]u *)
SigExpsFull[exps_, sym_Symbol, u_, m_] := 
  With[{Dexps=Table[D[t,sym],{t,exps}]},
    NestList[Outer[
     Assuming[sym>0,Integrate[# #2/.sym->t, {t, 0, sym}]] & , #, 
     Dexps] & , {1}, m]]/.sym->u

SigExps[exps_, sym_Symbol, u_, m_]:=
	First/@Rest[SigExpsFull[exps,sym,u,m]]

(** 2. Fns versions.
  A Fns (i.e. "functions") is a curve in R^d given in parametric form 
  as a list of functions.*)

(*If A is some structure of deeply nested lists, which all contain functions,
 then ApplyLowestLevel[A,x] is the same structure where each function
 has been replaced by the result of evaluating it at x. For example
 ApplyLowestLevel[{{Sqrt, {Cos}}, Sin}, 0] gives {{0,{1}},0} *)
ApplyLowestLevel[ff_, x_] := Last@Outer[Apply[#2, #1] &, {{x}}, ff, 1, Infinity]  
      
(*Full signature including initial 1 and an extra level of bracketing.*)
SigFnsFull[f_,u_,m_]:=
  ApplyLowestLevel[NestList[Outer[
                              Function[x, Integrate[#[t] Derivative[1][#2][t], {t, 0, x}]]& ,
                               #, f]& ,
                            {1& }, m],
                    u]
                    
(*SigFns[f,u,m] gets the signature of a Fns on [0,u] up to level m*)
SigFns[f_,u_,m_]:=
	First/@Rest[SigFnsFull[f,u,m]]
	
(*Plot[Evaluate[Simplify[VolFns[{Cos, Sin, Identity}, x]]], {x, 0, 30}]*)
(*f = {Cos, (((Sin[#])^2) &), #/2/Pi &}*)

(*Plot the parametric curve f on [0,u] where f is a list of functions*)
PlotFns[f_, u_] := 
 If[Length[f] == 2, ParametricPlot, ParametricPlot3D][
  Through[f[x]], {x, 0, u},ColorFunction->Function[{x,y,uu},Hue[0.75uu]]]
(*ShiftFns[f,7] is a Fns which on (0,1) is the same as f on (7,8)*)
(*ShiftFns[f_,by_]:={f[[1]][#-by]&,f[[2]][#-by]&,f[[3]][#-by]&}*)
ShiftFns[f_,by_]:=Table[With[{ff=ff},ff[#+by]&],{ff,f}]

(*If f and g are both Fnss, JoinFnss[f,g,2] is like f on [0,2] and 
on, say, [2,5] it is like g on [0,3], shifted to be continuous.*)
JoinFnss[f_,g_,break_]:=Table[With[{tt=tt,diff=tt[[1]][break]-tt[[2]][0]},
        Function[x,Piecewise[{{tt[[1]][x],x<break}},diff+tt[[2]][x-break]]]],
  {tt,Transpose[{f,g}]}]
EvaluateFns[f_,paramValue_]:=Through[f[paramValue]]
AddTimeToFns[f_]:=Append[f,Identity]

(*if p is a list of n points then Pts2Fn[p] is a Fns of the
 points linearly interpolated, on [0,n-1]. This is an inefficient thing to do! *)
Pts2Fns[points_]:=Table[ListInterpolation[points[[;;,i]],{{0,Length[points]-1}},
  InterpolationOrder->1],{i,Length[points[[1]]]}]


(** 3. Conversion between Fns and Exps *)

Exps2Fns[f_,sym_Symbol]:=Module[{ff},Table[Function@@{sym,ff},{ff,f}]]
Fns2Exps[f_,sym_Symbol]:=Through[f[sym]]
