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
  //	return static_cast<int> (std::round(std::pow(d,m)));
  return static_cast<int> (0.4+std::pow(d,m));
}

int calcSigTotalLength(int d, int m){
  if(d==1)
    return m;
  //	int p = static_cast<int> (std::round(std::pow(d,m)));
  int p = static_cast<int> (0.4 + std::pow(d,m));
  return d*(p-1)/(d-1);
}

class CalculatedSignature{
public:
  vector<vector<CalcSigNumeric>> m_data;

  template<typename Number>
  void sigOfSegment(int d, int m, const Number* segment){
    m_data.resize(m);
    m_data[0].assign(segment,segment+d);
    for(int level=2; level<=m; ++level){
      const auto& last = m_data[level-2];
      auto& s = m_data[level-1];
      s.assign(calcSigLevelLength(d,level),0);
      int i=0;
      for(auto l: last)
	for(auto p=segment; p<segment+d; ++p)
	  s[i++]=*p * l * (1.0/level);
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

  void writeOut(CalcSigNumeric* dest) const{
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


#endif
