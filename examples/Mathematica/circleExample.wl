(* ::Package:: *)

(*The signature of a circle up to level 6, reproduced here to save time.*)

(*sigOfCircle=SigFns[{Cos, Sin}, 2 Pi, 6]*)
sigOfCircle={{0, 0}, {{0, Pi}, {-Pi, 0}}, {{{0, -Pi}, {2*Pi, 0}}, 
  {{-Pi, 0}, {0, 0}}}, 
 {{{{0, (5*Pi)/8}, {(-15*Pi)/8, 0}}, 
   {{(15*Pi)/8, Pi^2/2}, {-Pi^2/2, Pi/8}}}, 
  {{{(-5*Pi)/8, -Pi^2/2}, {Pi^2/2, (-3*Pi)/8}}, 
   {{0, (3*Pi)/8}, {-Pi/8, 0}}}}, 
 {{{{{0, (-7*Pi)/24}, {(7*Pi)/6, 0}}, 
    {{(-7*Pi)/4, -Pi^2/2}, {Pi^2/2, -Pi/8}}}, 
   {{{(7*Pi)/6, Pi^2/2}, {0, (17*Pi)/24}}, 
    {{-Pi^2/2, (-11*Pi)/8}, {(11*Pi)/12, 0}}}}, 
  {{{{(-7*Pi)/24, 0}, {-Pi^2/2, -Pi/3}}, 
    {{Pi^2/2, (4*Pi)/3}, {(-11*Pi)/8, 0}}}, 
   {{{0, -Pi/3}, {(17*Pi)/24, 0}}, 
    {{-Pi/8, 0}, {0, 0}}}}}, 
 {{{{{{0, (7*Pi)/64}, {(-35*Pi)/64, 0}}, 
     {{(35*Pi)/32, (5*Pi^2)/16}, {(-5*Pi^2)/16, 
       (7*Pi)/96}}}, {{{(-35*Pi)/32, (-7*Pi^2)/16}, 
      {-Pi^2/16, (-21*Pi)/32}}, 
     {{Pi^2/2, (259*Pi)/192}, {(-175*Pi)/192, 0}}}}, 
   {{{{(35*Pi)/64, Pi^2/4}, {Pi^2/8, (119*Pi)/192}}, 
     {{-Pi^2/16, (-7*Pi)/4 + Pi^3/6}, 
      {(175*Pi)/96 - Pi^3/6, Pi^2/16}}}, 
    {{{(-5*Pi^2)/16, (7*Pi)/16 - Pi^3/6}, 
      {(Pi*(-175 + 16*Pi^2))/96, (-3*Pi^2)/16}}, 
     {{(175*Pi)/192, Pi^2/4}, {-Pi^2/8, Pi/192}}}}}, 
  {{{{{(-7*Pi)/64, -Pi^2/8}, {Pi^2/4, (-35*Pi)/192}}, 
     {{(-7*Pi^2)/16, (35*Pi)/96 - Pi^3/6}, 
      {(Pi*(-21 + 8*Pi^2))/48, -Pi^2/16}}}, 
    {{{(5*Pi^2)/16, (Pi*(-35 + 16*Pi^2))/96}, 
      {(7*Pi)/4 - Pi^3/6, (3*Pi^2)/16}}, 
     {{(-259*Pi)/192, (-3*Pi^2)/8}, 
      {Pi^2/4, (-5*Pi)/192}}}}, 
   {{{{0, (35*Pi)/192}, {(-119*Pi)/192, 0}}, 
     {{(21*Pi)/32, (3*Pi^2)/16}, {(-3*Pi^2)/16, 
       (5*Pi)/96}}}, {{{(-7*Pi)/96, -Pi^2/16}, 
      {Pi^2/16, (-5*Pi)/96}}, {{0, (5*Pi)/192}, 
      {-Pi/192, 0}}}}}}};

(*LogSigOfCircle_Lyndon=LeastSquares[Transpose@LogSigMatrix[2,6,GenerateLyndonBasis],
                Flatten[LogTensor[sigOfCircle]]]*)
logSigOfCircleLyndon={0, 0, Pi, -Pi, 0, (5*Pi)/8, 0, Pi/8, (-7*Pi)/24, 0, 
 -Pi/8, Pi/3, 0, 0, (7*Pi)/64, 0, 
 -Pi/144 + (-Pi^3/6 + (Pi*(-21 + 8*Pi^2))/48)/63 - 
  (5*(-Pi^3/6 + (Pi*(-175 + 16*Pi^2))/96))/126 - 
  (5*(-Pi^3/6 + (Pi*(-35 + 16*Pi^2))/96))/126, 
 (-1465*Pi)/3456 + 
  (37*(-Pi^3/6 + (Pi*(-21 + 8*Pi^2))/48))/1512 + 
  (-Pi^3/6 + (Pi*(-175 + 16*Pi^2))/96)/756 + 
  (-Pi^3/6 + (Pi*(-35 + 16*Pi^2))/96)/756, 0, 0, 0, 
 (-589*Pi)/3456 + (-Pi^3/6 + (Pi*(-21 + 8*Pi^2))/48)/
   1512 - (17*(-Pi^3/6 + (Pi*(-175 + 16*Pi^2))/96))/
   756 + (109*(-Pi^3/6 + (Pi*(-35 + 16*Pi^2))/96))/
   756, Pi/192};



(*Use this to see the above labelled with words:
TextGrid[Transpose[{Flatten[GenerateLyndonBasis[2,6],1],logSigOfCircleLyndon}]]*)

(*It's hard to see a pattern here, except that exactly when there's an even number of 2's
the value is 0*)

