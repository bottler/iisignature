#ifndef CALCSIGNATURE_H
#define CALCSIGNATURE_H

#include <cmath>
#include <vector>

using std::vector;

typedef float CalcSigNumeric;
//typedef vector<vector<CalcSigNumeric>> CalculatedSignature;

//Simple functions for calculating arbitrary signatures at runtime
//- perhaps slower to run but easier to use than the template version 

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


#endif
