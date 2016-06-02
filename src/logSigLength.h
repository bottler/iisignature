#ifndef LOGSIGLENGTH_H
#define LOGSIGLENGTH_H

//This file contains some simple number-theoretic constexpr functions,
//including calculating the length of the log signature.
//There is no reason to think these are efficient at runtime.

//Many of these functions assume their parameters are positive. 

namespace LogSigLength{

typedef long long Int;

//return the lowest factor of i no less than min
constexpr Int lowestFactorWithMin(Int i, Int min){
  return (min >= i || i % min ==0) ? min : lowestFactorWithMin(i, min+1);
}

//return the lowest prime factor of i
//crude, will be wrong if i<=1
constexpr Int lowestFactor(Int i){
  return i<=3 ? i : 
           i%2 ==0 ? 2 :
              lowestFactorWithMin(i,3); 
}

//crude, will be wrong if i<=1
constexpr bool isPrime(Int i){
  return i==lowestFactor(i);
}

//return the highest j such that p^j divides i
constexpr Int primeToHowMuch(Int i, Int p){
  return (i%p) ? 0 : 1+primeToHowMuch(i/p,p);
}

//return i with all factors of p divided out
constexpr Int removeFactor(Int i, Int p){
  return i%p ? i : removeFactor(i/p, p);
}

//internal use
constexpr Int mobiusFactor(Int i, Int p){
  return primeToHowMuch(i,p) > 1 ? 0 : -1;
}

//internal use
constexpr Int mobiusWithMin(Int i, Int min){
  return i<=1 ? 1 : 
    1==removeFactor(i,lowestFactorWithMin(i,min)) ? mobiusFactor(i, lowestFactorWithMin(i,min)) :
      mobiusFactor(i, lowestFactorWithMin(i,min)) * mobiusWithMin(removeFactor(i,lowestFactorWithMin(i,min)),1+min);
}

//the mobius function
constexpr Int mobius(Int i){
  return mobiusWithMin(i,2);  
}

//return a^b
constexpr Int power(Int a, Int b){
  return b==0 ? 1 : a*power(a,b-1);
}

//internal use
constexpr Int necklaceSummand(Int d, Int m, Int divisor){
  return mobius(m/divisor)*power(d,divisor);
}

//sum of necklaceSummand for divisor ranging over all factors of m no less than from
//internal use
constexpr Int necklaceSum(Int d, Int m, Int from){
  return from == m ? necklaceSummand(d,m,m) : 
    necklaceSummand(d,m,from) + necklaceSum(d,m,lowestFactorWithMin(m,from+1));
}

//The number of necklaces on d letters with length m
constexpr Int countNecklacesLengthM(Int d, Int m){
  return necklaceSum(d,m,1)/m;
}

//The number of necklaces on d letters with length m or less
//= The length of the log signature in d dimensions up to level m
constexpr Int countNecklacesUptoLengthM(Int d, Int m){
  return countNecklacesLengthM(d,m)
           + ( m==1 ? 0 : countNecklacesUptoLengthM(d,m-1) );
}

//the length of the signature in d dimensions up to level m
//(excluding the constant 1, of course)
constexpr Int sigLength(Int d, Int m){
  return (d*(power(d,m)-1))/(d-1);
}

}//namespace LogSigLength

#endif
