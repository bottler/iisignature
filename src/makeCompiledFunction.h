#ifndef MAKECOMPILEDFUNCTION_H
#define MAKECOMPILEDFUNCTION_H
#include<vector>
#include<iostream>
#include <stdexcept>
#include<algorithm>
#include<map>
#include<cstdint>

#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#else
# include <sys/mman.h>
#endif

//#include "spacedVector.h"

//Mem is a memory buffer in which you write a function in machine code.
class Mem{
public:
  template<typename T> T getFn(){
    return reinterpret_cast<T> (m_buf);
  }
  Mem(size_t s):m_size(s){
    //we may want to use space at the beginning for double constants
#ifdef _WIN32
    m_p = m_buf = VirtualAllocEx( GetCurrentProcess(), 0, s, MEM_COMMIT, PAGE_EXECUTE_READWRITE );
#else
    m_p = m_buf =(unsigned char*) mmap(0,s,PROT_READ|PROT_EXEC|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
#endif
    //    std::cout<<"Align "<<((size_t)m_p)%1024<<"\n"; //Looks like this is always 0, yay!
//on windows, use 
  }
  ~Mem(){
#ifdef _WIN32
    bool i = !VirtualFreeEx(GetCurrentProcess(),m_buf,0,MEM_DECOMMIT);
#else
    int i = munmap(m_buf, m_size);
#endif
    if(i)
      std::cout<<"problem freeing buf"<<std::endl;
  }
  void push(unsigned char p){
    if(m_buf + m_size <= m_p)
      throw std::runtime_error("overflowed buffer in push()");
    *(m_p++) = p;
  }
  void push(unsigned char p1, unsigned char p2){
    push(p1); push(p2);
  }
  void push(unsigned char p1, unsigned char p2, unsigned char p3){
    push(p1); push(p2); push(p3);
  }
  void push(unsigned char p1, unsigned char p2, unsigned char p3, unsigned char p4){
    push(p1); push(p2); push(p3); push(p4);
  }
  void pushLittleEndian(uint64_t a){
    for(int i=0; i<8; ++i){
      push(a%256);
      a=a/256;
    }   
  }
  void pushLittleEndian(uint32_t a){
    for(int i=0; i<4; ++i){
      push(a%256);
      a=a/256;
    }   
  }
  size_t capacity() const {return m_size;}
  size_t used() const {return m_p-m_buf;}
private:
  unsigned char* m_buf;
  unsigned char* m_p;
  size_t m_size;
};

//We have a function with type void(double* a, double* b, double* t). InputArr represents one of the arguments and the register it is in.
enum class InputArr {A, B, T, C}; //C not a real arg, won't exist in FunctionData
typedef std::pair<InputArr, int> InputPos;
struct FunctionData{
  std::vector<std::pair<InputPos,InputPos>> m_formingT;
  std::vector<double> m_constants;
  //std::vector<double,boost::alignment::aligned_allocator<double,64>> m_constants;
  struct LineData {int m_lhs_offset, m_rhs_offset, m_const_offset; bool m_negative;};
  std::vector<LineData> m_lines;
  size_t m_length_of_b; //to add on at end
};

//The following function is the function which Maker creates (except this function takes d as a parameter, and that function takes t as a parameter).
void slowExplicitFunction(double* a, const double* b, const FunctionData& d){
  std::vector<double> t(d.m_formingT.size());
  for(size_t i=0; i<d.m_formingT.size(); ++i){
    double lhs, rhs;
    switch(d.m_formingT[i].first.first){
    case InputArr::A:
      lhs = a[d.m_formingT[i].first.second];
      break;
    case InputArr::B:
      lhs = b[d.m_formingT[i].first.second];
      break;
    default:
      lhs = t[d.m_formingT[i].first.second];
    }    
    switch(d.m_formingT[i].second.first){
    case InputArr::A:
      rhs = a[d.m_formingT[i].second.second];
      break;
    case InputArr::B:
      rhs = b[d.m_formingT[i].second.second];
      break;
    default:
      rhs = t[d.m_formingT[i].second.second];
    }    
    t[i] = lhs * rhs;
  }
  for(auto l : d.m_lines){
    a[l.m_lhs_offset] += (l.m_negative ? -1 : 1) * t[l.m_rhs_offset] * d.m_constants[l.m_const_offset];
  }
  for(size_t i=0; i<d.m_length_of_b; ++i)
    a[i]+=b[i];
}


//To use an 8 bit offset (rather than 32) into a vector<double> 
//you waste 3 bits, 1 is the sign, so you can move +-16 or so
//could be better to use floats

//Todo: this is all based on the usual x64 instructions. We could try using FMA3, FMA4 and AVX - Compare compiling things with -march=native -O3. also look at vectorising. add a processor check.
//http://www.agner.org/optimize/
//false dependency problems https://stackoverflow.com/questions/11177137/why-do-most-x64-instructions-zero-the-upper-part-of-a-32-bit-register

struct Maker{

  static unsigned char getRegNumber(InputArr array){
#ifdef _WIN32
    unsigned char start = array == InputArr::A ? 1 : //RCX
                          array == InputArr::B ? 2 : //RDX
	    	          array == InputArr::T ? 7 : //let's use RDI
			  6;                         //RSI
#else
    unsigned char start = array == InputArr::A ? 7 : //RDI
                          array == InputArr::B ? 6 : //RSI
			  array == InputArr::T ? 2 : //RDX
			  1;                         //RCX
#endif
    return start;
  }


    //pushes to m a ModR/M for XMM(k) and i.first[i.second] where i.first is register containing a double*
    //k is between 0 and 7
    //pushes at most 5 bytes
  void xmmkWithRegOffset(Mem& m, const InputPos& i, unsigned char k = 0){
    unsigned char start = getRegNumber(i.first);
    //  const bool squashBads = true; //just to test
    start += k*8;
    if(i.second<0)
      throw std::runtime_error("What?");  
    if(i.second==0)
      m.push(start);
    else if(i.second<16){
      m.push(0x40+start);
      m.push(8*i.second);
      ++m_singles;
    }else{
      ++(i.second<32 ? m_nearlies : m_bads);
      m_bads_by_register[(int)(i.first)]++;
      m.push(0x80+start);
      uint32_t ii = 8*i.second;
      m.pushLittleEndian(ii);
    }
  }

  //We are making void form_t(const double* a, const double* b, double* t) with no ret
  //pushes at most m_formingT.size() * 24 bytes
  void make_form_t(Mem& m, const FunctionData& d){
    using std::make_pair;
    for(size_t i=0; i<d.m_formingT.size(); ++i){
      //First movsd the left
      m.push(0xf2, 0x0f, 0x10);
      xmmkWithRegOffset(m,d.m_formingT[i].first);
      //now mulsd
      m.push(0xf2, 0x0f, 0x59);
      if(d.m_formingT[i].first == d.m_formingT[i].second){
        //just square it
        m.push(0xc0);
      }
      else
	xmmkWithRegOffset(m,d.m_formingT[i].second);
      //Now movsd back
      m.push(0xf2, 0x0f, 0x11);
      xmmkWithRegOffset(m,make_pair(InputArr::T, i));
    }  
  }

  //we are making void main_multiplies(double* a, const double* b, const double* t) with no ret
  //pushes at most d.m_lines.size() * 36 bytes
  void make_main_multiplies(Mem& m, const FunctionData& d){
    using std::make_pair;
    //for the moment, assume constants in RCX (=4th pointer arg) = InputArr::C
    unsigned char base_xmm = 0;
    for(size_t idx = 0; idx<d.m_lines.size(); ++idx){
      const auto& l = d.m_lines[idx];
      //first calculate RHS
      //movsd the constant
      m.push(0xf2, 0x0f, 0x10);
      xmmkWithRegOffset(m,make_pair(InputArr::C, l.m_const_offset),base_xmm);
      //xmmkWithRegOffset(m,make_pair(InputArr::T, l.m_rhs_offset),base_xmm);//FAKE
      //mulsd the t offset
      m.push(0xf2, 0x0f, 0x59);
      xmmkWithRegOffset(m,make_pair(InputArr::T, l.m_rhs_offset),base_xmm);
      //xmmkWithRegOffset(m,make_pair(InputArr::T, 3),base_xmm);//FAKE
      if(true) //try to amalgamate with subsequent lines before writing to memory.
	while(l.m_lhs_offset==d.m_lines[idx+1].m_lhs_offset){
          ++idx;
	  const auto& l2 = d.m_lines[idx];
	  //Load next into xmm1
	  m.push(0xf2, 0x0f, 0x10);
	  xmmkWithRegOffset(m,make_pair(InputArr::C, l2.m_const_offset),1+base_xmm);
	  //mulsd the t offset
	  m.push(0xf2, 0x0f, 0x59);
	  xmmkWithRegOffset(m,make_pair(InputArr::T, l2.m_rhs_offset),1+base_xmm);
	  //add it back
	  if(l.m_negative == l2.m_negative)
	    //add xmm(1+base_xmm) to xmm(base_xmm)
	    m.push(0xf2, 0x0f, 0x58, 0xc1+9*base_xmm);
	  else
	    //sub xmm(1+base_xmm) from xmm(base_xmm)
	    m.push(0xf2, 0x0f, 0x5c, 0xc1+9*base_xmm);	
        }
      if(!l.m_negative){
        //addsd
        m.push(0xf2, 0x0f, 0x58);
	xmmkWithRegOffset(m,make_pair(InputArr::A, l.m_lhs_offset),base_xmm);
	//Now movsd back
	m.push(0xf2, 0x0f, 0x11);
	xmmkWithRegOffset(m,make_pair(InputArr::A, l.m_lhs_offset),base_xmm);
      }else{
        //Load the LHS into xmm1
        m.push(0xf2, 0x0f, 0x10);
	xmmkWithRegOffset(m,make_pair(InputArr::A, l.m_lhs_offset),1+base_xmm);
	//subsd xmm(base_xmm) from xmm(1+base_xmm)
	m.push(0xf2, 0x0f, 0x5c, 0xc8+9*base_xmm);
	//Now movsd back
	m.push(0xf2, 0x0f, 0x11);
	xmmkWithRegOffset(m,make_pair(InputArr::A, l.m_lhs_offset),1+base_xmm);
      }
      //does it make any difference whether we always use the same pair of registers?
      base_xmm = base_xmm+2;
#ifdef _WIN32
      if(base_xmm==6) //xmm6 and greater are nonvolatile on windows. For the moment, we just don't touch them.
	base_xmm = 0;
#else
      if(base_xmm==8)
	base_xmm = 0;
#endif
    }
  }

  //pushes 8*d.m_length_of_b bytes
  void make_final_adds(Mem& m, const FunctionData& d){
    using std::make_pair;
    const auto len = d.m_length_of_b;
    for(size_t i=0; i<len; ++i){
      m.push(0xf2, 0x0f, 0x10);//movsd from b[len-i] to xmm0
      xmmkWithRegOffset(m,make_pair(InputArr::B, len-1-i));
      m.push(0xf2, 0x0f, 0x58);//addsd from A[len-i] to xmm0
      xmmkWithRegOffset(m,make_pair(InputArr::A, len-1-i));
      m.push(0xf2, 0x0f, 0x11);//movsd to A[len-i]
      xmmkWithRegOffset(m,make_pair(InputArr::A, len-1-i));
    }
  }

  //pushes at most  d.m_lines.size() * 36 + m_formingT.size() * 24 + 3 bytes, 7 more on windows
  void make(Mem& m, const FunctionData& d)
  {

#ifdef _WIN32
    //PUSH the value of the nonvolatile register we wanna use for constants to the stack
    m.push(0x50 + getRegNumber(InputArr::C));
    m.push(0x50 + getRegNumber(InputArr::T));
    //MOV the third argument, T, from R8 where windows sticks it, to our preferred register
    //this saves using a REX byte every time we access it!
    //this REX - 4c - has a bit for 64bit and one to add to the 0 for R8
    m.push(0x4c, 0x89, getRegNumber(InputArr::T)); 
#endif

    make_form_t(m,d);
    
    //Load the constants pointer into a register
    //0x48 is a REX to say it's a 64 bit operand
    m.push(0x48, 0xb8 + getRegNumber(InputArr::C));
    m.pushLittleEndian((uint64_t)d.m_constants.data());
    
    make_main_multiplies(m,d);
    make_final_adds(m,d);
    
#ifdef _WIN32
    //POP the values of the nonvolatile registers we pushed
    m.push(0x58 + getRegNumber(InputArr::T));
    m.push(0x58 + getRegNumber(InputArr::C));
#endif
    m.push(0xc3);//retq
    
    if(0){
      std::cout<<d.m_length_of_b<<"\n";
      std::cout<<d.m_constants.size()<<"\n";
      std::cout<<m_singles<<","<<m_bads<<","<<m_nearlies<<"\n";
      for(auto i : m_bads_by_register)
	std::cout<<":"<<i.first<<":"<<i.second;
      std::cout<<"\n"<<m.used() <<" out of "<<m.capacity()<<std::endl;
    }
  }

  int m_singles=0;
  int m_bads=0;
  int m_nearlies=0;
  std::map<int,int> m_bads_by_register;

};

struct FunctionRunner{
  const FunctionData& m_d;
  std::vector<double> m_t; //working space
  ///vector<double,boost::alignment::aligned_allocator<double,64>> m_t; //makes no difference?
  Mem m_m;
  FunctionRunner(FunctionData& d)
  : m_d(d),
    m_m(d.m_lines.size()*36+d.m_formingT.size()*24+d.m_length_of_b*8+10)
  {
    //d.m_lines.size()//126
    //This sorting by rhs_offset helps a lot, even if  rhs_offset is never touched
    if(true)
      std::sort(d.m_lines.begin(), d.m_lines.end(),[](
						    const FunctionData::LineData& a,
						    const FunctionData::LineData& b){
		return a.m_rhs_offset<b.m_rhs_offset
				      //||(a.m_rhs_offset==b.m_rhs_offset && a.m_lhs_offset>b.m_lhs_offset)
				      //||(a.m_rhs_offset==b.m_rhs_offset && a.m_const_offset>b.m_const_offset)
		  ;
		      
	      });
    
    //*/
    //perhaps we want to separate the lhs offset as much as possible
    //    for(auto& i : d.m_lines)
    //  i.m_lhs_offset=4;
    if(true)
    {
      std::stable_sort(d.m_lines.begin(), d.m_lines.end(),[](
						    const FunctionData::LineData& a,
						    const FunctionData::LineData& b){
		return a.m_lhs_offset<b.m_lhs_offset;		      
	      });
      /*
      const int gap = 3;
      std::stable_sort(d.m_lines.begin(), d.m_lines.end(),[](
						    const FunctionData::LineData& a,
						    const FunctionData::LineData& b){
		return a.m_lhs_offset%gap < b.m_lhs_offset%gap;		      
	      });
      */
    }
    if(false)
    for(size_t i=0; i<d.m_lines.size(); ++i){
      d.m_lines[i].m_lhs_offset = i%14;
      d.m_lines[i].m_rhs_offset = 4;
      d.m_lines[i].m_const_offset = 3;
    }
    m_t.resize(d.m_formingT.size());
    //std::cout<<"Align t "<<((size_t)m_t.data())%1024<<"\n";
    //std::cout<<"Align c "<<((size_t)m_d.m_constants.data())%1024<<"\n";
    //d.m_formingT.clear();
    Maker maker;
    maker.make(m_m,m_d);
  }
  void go(double* a, const double* b){
    typedef void (*my_fn_type)(double*, const double*, double*);
    auto f = m_m.getFn<my_fn_type>();
    f(a,b,m_t.data());
  }
};

#endif
