#ifndef CALCSIGNATURE_H
#define CALCSIGNATURE_H

#include <cmath>
#include <vector>

int calcSigLevelLength(int d, int m){
  //    return static_cast<int> (std::round(std::pow(d,m)));
  return static_cast<int> (0.4+std::pow(d,m));
}

int calcSigTotalLength(int d, int m){
  if(d==1)
    return m;
  //    int p = static_cast<int> (std::round(std::pow(d,m)));
  int p = static_cast<int> (0.4 + std::pow(d,m));
  return d*(p-1)/(d-1);
}

namespace CalcSignature{
  using std::vector;

  typedef float CalcSigNumeric;
  //typedef vector<vector<CalcSigNumeric>> CalculatedSignature;

  //Simple functions for calculating arbitrary signatures at runtime
  //- perhaps slower to run but easier to use than the template version 

 
  class CalculatedSignature{
  public:
    vector<vector<CalcSigNumeric>> m_data;
    
    template<typename Number>
    void sigOfSegment(int d, int m, const Number* segment){
      m_data.resize(m);
      auto& first = m_data[0];
      first.resize(d);
      for(int i=0; i<d; ++i)
        first[i]=(CalcSigNumeric)segment[i];
      for(int level=2; level<=m; ++level){
        const auto& last = m_data[level-2];
        auto& s = m_data[level-1];
        s.assign(calcSigLevelLength(d,level),0);
        int i=0;
        for(auto l: last)
          for(auto p=segment; p<segment+d; ++p)
            s[i++]=(CalcSigNumeric)(*p * l * (1.0/level));
      }
    }
    
    void sigOfNothing(int d, int m){
      m_data.resize(m);
      m_data[0].assign(d,0);
      for(int level=2; level<=m; ++level){
        auto& s = m_data[level-1];
        s.assign(calcSigLevelLength(d,level),0);
      }
    }
  
    //if a is the signature of path A, b of B, then
    //a.concatenateWith(d,m,b) makes a be the signature of the concatenated path AB
    //This is also the (concatenation) product of the elements a and b in the tensor algebra.
    void concatenateWith(int d, int m, const CalculatedSignature& other){
      for(int level=m; level>0; --level){
        for(int mylevel=level-1; mylevel>0; --mylevel){
          int otherlevel=level-mylevel;
          auto& oth = other.m_data[otherlevel-1];
          for(auto dest=m_data[level-1].begin(), 
                my =m_data[mylevel-1].begin(),
                myE=m_data[mylevel-1].end(); my!=myE; ++my){
            for(const CalcSigNumeric& d : oth){
              *(dest++) += d * *my;
            }
          }
        }
        auto source =other.m_data[level-1].begin();
        for(auto dest=m_data[level-1].begin(),
              e=m_data[level-1].end();
            dest!=e;)
          *(dest++) += *(source++);
        
      }
    }
    void swap(CalculatedSignature& other){
      m_data.swap(other.m_data);
    }

    void multiplyByConstant(CalcSigNumeric c){
      for(auto& a: m_data)
        for(auto& b:a)
          b*=c;    
    }
    
    template<typename Numeric>
    void writeOut(Numeric* dest) const{
      for(auto& a: m_data)
        for(auto& b:a)
          *(dest++)=b;
    }
    void writeOutExceptLasts(CalcSigNumeric* dest) const{
      for(auto& a: m_data)
        for(auto i = a.begin(), e=a.end()-1; i!=e; ++i)
          *(dest++)=*i;
    }
    
  };

  //This also calculates the concatenation product in the tensor algebra, 
  //but in the case where we assume 0 instead of 1 in the zeroth level.
  //It is not in-place
  CalculatedSignature concatenateWith_zeroFirstLevel(int d, int m,
                                                     const CalculatedSignature& a,
                                                     const CalculatedSignature& b){
    CalculatedSignature out;
    out.sigOfNothing(d,m);
    for(int level=m; level>0; --level){
      for(int alevel=level-1; alevel>0; --alevel){
        int blevel=level-alevel;
        auto& aa = a.m_data[alevel-1];
        auto& bb = b.m_data[blevel-1];
        auto dest=out.m_data[level-1].begin();
        for(const CalcSigNumeric& c : aa){
          for(const CalcSigNumeric& d : bb){
            *(dest++) += d * c;
          }
        }
      }
    }
    return out;
  }
  
  void logTensor(CalculatedSignature& s){
    const int m = (int)s.m_data.size();
    const int d = (int)s.m_data[0].size();
    vector<CalculatedSignature> powers;
    powers.reserve(m);
    powers.push_back(s);
    for(int power = 2; power<=m; ++power){
      powers.push_back(concatenateWith_zeroFirstLevel(d,m,powers.back(),s));
    }
    bool neg = true;
    for(int power = 2; power<=m; ++power){
      powers[power-1].multiplyByConstant((CalcSigNumeric)(neg ? (-1.0/power) : (1.0/power)));
      neg = !neg;
    }
    for(int power = 2; power<=m; ++power){
      for(int level=0; level<m; ++level)
        for(size_t i=0; i<s.m_data[level].size(); ++i)
          s.m_data[level][i] += powers[power-1].m_data[level][i];
    }  
  }
}

namespace TotalDerivativeSignature{
  using std::vector;
  using Number = double;
  //represents a calculated value and its derivative wrt a set of inputs
  struct DiffVariable{
    Number m_value;
    std::vector<Number> m_derivs;//these always have the same length.
  };
  
  //x = x+y
  void sumInPlace(DiffVariable& x, const DiffVariable& y){
    for(size_t i=0; i<x.m_derivs.size(); ++i){
      x.m_derivs[i] += y.m_derivs[i];
    }
    x.m_value += y.m_value;
  }
  //x = x*y
  void multiplyInPlace(DiffVariable& x, const DiffVariable& y){
    for(size_t i=0; i<x.m_derivs.size(); ++i){
      x.m_derivs[i] = x.m_derivs[i]*y.m_value + x.m_value*y.m_derivs[i];
    }
    x.m_value *= y.m_value;
  }
  //x += y*z
  void accumulateMultiply(DiffVariable& x, const DiffVariable& y, const DiffVariable& z){
    auto s = x.m_derivs.size();
    for(size_t i=0; i<s; ++i){
      x.m_derivs[i] += z.m_derivs[i]*y.m_value + z.m_value*y.m_derivs[i];
    }
    x.m_value += y.m_value * z.m_value;
  }
  //returns x*y*scalar
  DiffVariable multiply(const DiffVariable& x, const DiffVariable& y, 
                        Number fixedscalar){
    auto s = x.m_derivs.size(); 
    DiffVariable out;
    out.m_derivs.resize(s);
    for(size_t i=0; i<s; ++i){
      out.m_derivs[i] = (x.m_derivs[i]*y.m_value + x.m_value*y.m_derivs[i])*fixedscalar;
    }
    out.m_value = x.m_value * y.m_value * fixedscalar;
    return out;
  }
  
  struct InputSegment{
    std::vector<DiffVariable> m_segment;
  };
  
  struct Signature{
    vector<vector<DiffVariable>> m_data;
    
    void sigOfSegment(int d, int m, const InputSegment& segment){
      m_data.resize(m);
      auto& first = m_data[0];
      first = segment.m_segment;
      for(int level=2; level<=m; ++level){
        const auto& last = m_data[level-2];
        auto& s = m_data[level-1];
        s.resize(calcSigLevelLength(d,level));
        int i=0;
        for(const auto& l: last)
          for(const auto& p : segment.m_segment)
            s[i++]=multiply(p,l,(1.0f/level));
      }
    }
    //if a is the signature of path A, b of B, then
    //a.concatenateWith(d,m,b) makes a be the signature of the concatenated path AB
    //This is also the (concatenation) product of the elements a and b in the tensor algebra.
    void concatenateWith(int d, int m, const Signature& other){
      for(int level=m; level>0; --level){
        for(int mylevel=level-1; mylevel>0; --mylevel){
          int otherlevel=level-mylevel;
          auto& oth = other.m_data[otherlevel-1];
          for(auto dest=m_data[level-1].begin(),
                my =m_data[mylevel-1].begin(),
                myE=m_data[mylevel-1].end(); my!=myE; ++my){
            for(const auto& d : oth){
              accumulateMultiply(*(dest++), d,*my);
            }
          }
        }
        auto source =other.m_data[level-1].begin();
        for(auto dest=m_data[level-1].begin(),
              e=m_data[level-1].end();
            dest!=e;)
          sumInPlace(*(dest++),*(source++));
      }
    }
    void swap(Signature& other){
      m_data.swap(other.m_data);
    }
  };

  //makes s be input(index+1,i)-input(index,i) for i in [0,d)
  void assignSegment(InputSegment& s, const Number* input, size_t inputSize, size_t index, int d){
    s.m_segment.resize(d);
    for(int i=0; i!=d; ++i){
      s.m_segment[i].m_value=input[(index+1)*d+i] - input[index*d+i];
      s.m_segment[i].m_derivs.assign(inputSize,0.0f);
      s.m_segment[i].m_derivs[(index+1)*d+i]=1.0f;
      s.m_segment[i].m_derivs[index*d+i]=-1.0f;
    }
  }
  //input is n*d
  //out is assumed to be readily sized inputSize*totalSiglength
  void sigJacobian(const Number* input, 
                   int n, int d, int m, float* out){
    size_t inputSize = n*d;
    size_t totalSigLength = calcSigTotalLength(d,m);
    if (n==1)
      for(size_t i=0; i<inputSize*totalSigLength; ++i)
        out[i]=0.0f;
    Signature s1, s2;
    InputSegment is;
    for(int index=0; index+1<n; ++index){
      assignSegment(is,input, inputSize, index, d);
      s1.sigOfSegment(d,m,is);
      if(index==0)
        s2.swap(s1);
      else
        s2.concatenateWith(d,m,s1);
    }
    int i=0;
    for(auto& level : s2.m_data)
      for(auto& value : level){
        for(size_t j=0; j<inputSize; ++j)
          out[i+j*totalSigLength]=value.m_derivs[j];
        ++i;
      }
  }
}

namespace BackwardDerivativeSignature{
  //functions to pass a derivative of some scalar F back through a signature calculation
  using std::vector;
  using Number = double;

  //Simple functions for calculating arbitrary signatures at runtime
  //- perhaps slower to run but easier to use than the template version 

 
  class Signature{
  public:
    vector<vector<Number>> m_data;
    
    template<typename Num>
    void sigOfSegment(int d, int m, const Num* segment){
      m_data.resize(m);
      auto& first = m_data[0];
      first.resize(d);
      for(int i=0; i<d; ++i)
        first[i]=(Number)segment[i];
      for(int level=2; level<=m; ++level){
        const auto& last = m_data[level-2];
        auto& s = m_data[level-1];
        s.assign(calcSigLevelLength(d,level),0);
        int i=0;
        for(auto l: last)
          for(auto p=segment; p<segment+d; ++p)
            s[i++]=(Number)(*p * l * (1.0/level));
      }
    }
  
    //if a is the signature of path A, b of B, then
    //a.concatenateWith(d,m,b) makes a be the signature of the concatenated path AB
    //This is also the (concatenation) product of the elements a and b in the tensor algebra.
    void concatenateWith(int d, int m, const Signature& other){
      for(int level=m; level>0; --level){
        for(int mylevel=level-1; mylevel>0; --mylevel){
          int otherlevel=level-mylevel;
          auto& oth = other.m_data[otherlevel-1];
          for(auto dest=m_data[level-1].begin(), 
                my =m_data[mylevel-1].begin(),
                myE=m_data[mylevel-1].end(); my!=myE; ++my){
            for(const Number& d : oth){
              *(dest++) += d * *my;
            }
          }
        }
        auto source =other.m_data[level-1].begin();
        for(auto dest=m_data[level-1].begin(),
              e=m_data[level-1].end();
            dest!=e;)
          *(dest++) += *(source++);
      }
    }
    //if a is the signature of the concatenated path AB, b of the straight line segment B, then
    //a.unconcatenateWith(d,m,b) makes a be the signature of A
    //this is like concatenateWith except that other is taken to be negative in its odd levels
    //which depends critically on the fact that other is the signature of a straight line
    void unconcatenateWith(int d, int m, const Signature& other){
      for(int level=m; level>0; --level){
        for(int mylevel=level-1; mylevel>0; --mylevel){
          int otherlevel=level-mylevel;
          const float factor = ((otherlevel % 2) ? -1.0f : 1.0f);
          auto& oth = other.m_data[otherlevel-1];
          for(auto dest=m_data[level-1].begin(), 
                my =m_data[mylevel-1].begin(),
                myE=m_data[mylevel-1].end(); my!=myE; ++my){
            for(const Number& d : oth){
              *(dest++) += d * *my * factor;
            }
          }
        }
        const float factor = ((level % 2) ? -1.0f : 1.0f);
        auto source =other.m_data[level-1].begin();
        for(auto dest=m_data[level-1].begin(),
              e=m_data[level-1].end();
            dest!=e;)
          *(dest++) += *(source++) * factor;       
      }
    }
    void swap(Signature& other){
      m_data.swap(other.m_data);
    }
    void fromRaw(int d, int m, const Number* raw){
      m_data.resize(m);
      size_t levelLength = d;
      for(int level=1; level<=m; ++level){
        const Number* end = raw+levelLength;
        auto& s = m_data[level-1];
        s.assign(raw,end);
        raw=end;
        levelLength *= d;
      }
    }
  };

  //let A and B be paths with signatures sig(A), sig(B) and sig(AB) the sig of the concatenated path, 
  // where B is straight. Let F be some scalar
  //input: b is sig(B), a is sig(AB) and ww is dF/d(sig(AB))
  //output: a is sig(A), bb is dF/d(sig(B)), ww is dF/d(sig(AB))
  void backConcatenate(int d, int m, Signature& a, const Signature& b, Signature& ww, Signature& bb){
    a.unconcatenateWith(d,m,b);
    bb=ww;
    //in this block, we only modify bb
    for(int level=m; level>0; --level){
      for(int mylevel=level-1; mylevel>0; --mylevel){
        int otherlevel=level-mylevel;
        auto& oth = bb.m_data[otherlevel-1];
        for(auto dest=ww.m_data[level-1].begin(), 
              my =a.m_data[mylevel-1].begin(),
              myE=a.m_data[mylevel-1].end(); my!=myE; ++my){
          for(Number& d : oth){
            d += *(dest++) * *my;
          }
        }
      }
    }
    //in this block, we only modify ww. 
    //The level which we modify increases, and the level we read is always higher
    //so note the loops are different but equivalent to all the others
    for(int mylevel=1; mylevel<m; ++mylevel){
      for(int level=mylevel+1; level<=m; ++level){
        int otherlevel=level-mylevel;
        auto& oth = b.m_data[otherlevel-1];
        for(auto dest=ww.m_data[level-1].begin(), 
              my =ww.m_data[mylevel-1].begin(),
              myE=ww.m_data[mylevel-1].end(); my!=myE; ++my){
          for(const Number& d : oth){
            *my += *(dest++) * d;
          }
        }
      }
    }
  }

  using OutputNumber = float;

  //if X is a line segment with signature x and s is dF/d(sig(X)) for some scalar F,
  //then this function increments pos and decrements neg by dF/d(displacement of X)
  //and leaves s in a meaningless state
  void backToSegment(int d, int m, const Signature& x, Signature& s, OutputNumber* pos, OutputNumber* neg){
    const auto& segment = x.m_data[0];
    auto& dSegment = s.m_data[0];
    for(int level = m; level>1; --level){
      auto i = s.m_data[level-1].begin();
      for(size_t j=0; j<s.m_data[level-2].size(); ++j)
        for(int dd=0; dd<d; ++dd, ++i){
          s.m_data[level-2][j] += segment[dd] * (1.0/level) * *i;
          dSegment[dd] += x.m_data[level-2][j] * (1.0/level) * *i;
        }
    }
    auto& source = s.m_data[0];
    for(int i = 0; i<d; ++i){
      const Number n = source[i];
      pos[i] += n;
      neg[i] -= n;
    }
  }


  void calcSignature(int d, int m, int lengthOfPath, const Number* data, Signature& s2){
    Signature s1;
    vector<Number> displacement(d);
    for(int i=1; i<lengthOfPath; ++i){
      for(int j=0;j<d; ++j)
        displacement[j]=data[i*d+j]-data[(i-1)*d+j];
      s1.sigOfSegment(d,m,&displacement[0]);
      if(i==1)
        s2.swap(s1);
      else
        s2.concatenateWith(d,m,s1);
    }    
  }

  //path is a (lengthOfPath)xd path, derivs is dF/d(sig(path)) of length siglength(d,m), 
  //output is (lengthOfPath)xd
  //this function just increments output[i,j] by dF/d(path[i,j])
  void sigBackwards(int d, int m, int lengthOfPath, const Number* path, 
                    const Number* derivs, OutputNumber* output){
    if(lengthOfPath<2)
      return;
    Signature allSigDerivs, allSig, localDerivs,segmentSig;
    calcSignature(d,m,lengthOfPath, path, allSig);
    allSigDerivs.fromRaw(d,m,derivs);
    vector<Number> displacement(d);
    for(int i=lengthOfPath-1; i>0; --i){
      for(int j=0; j<d; ++j)
        displacement[j]=path[i*d+j]-path[(i-1)*d+j];
      segmentSig.sigOfSegment(d,m,&displacement[0]);
      backConcatenate(d,m,allSig,segmentSig,allSigDerivs,localDerivs);
      backToSegment(d,m,segmentSig,localDerivs,output+i*d,output+(i-1)*d);
    }
  }
}

#endif
