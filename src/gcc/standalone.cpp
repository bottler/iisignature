#include "bch.hpp"
#include "logsig.hpp"
#include<fstream>

using std::vector;
using std::cout;
using std::endl;

void doNothing(){}

void testFreq(){
  BasisElt lw4(4);
  BasisElt lw5(5);
  BasisElt lw6(6);
  BasisElt l (lw4, BasisElt(lw4, BasisElt(BasisElt(lw6,lw4), BasisElt(lw4, BasisElt(lw5,lw6)))));
  auto f= getLetterFrequencies(&l);
  for(size_t i : f)
    std::cout <<i<<"\n";
}

int main(){
  //int i;std::cin>>i;
  //  printListOfLyndonWords(2,2);
  const char* file = "../../iisignature_data/bchLyndon20.dat";
  std::ifstream stream (file);
  auto s = new std::string(std::istreambuf_iterator<char>(stream), {});
  g_bchLyndon20_dat = s->c_str();

#if 0
  auto aa = ReadBCH::read();
  const auto &a = aa.m_rows;
  std::cout<<a.size()<<"\n";
  auto r = a.back();
  std::cout<<r.m_left<<" "<<r.m_right<<" "<<r.m_coeff<<"\n";
  for(int i : aa.m_totalLengths)
    std::cout<<i<<" ";
#endif
  //calcFla(2,4,doNothing);
  LogSigFunction lsf(LieBasis::Lyndon);
  vector<double> a{0,1}, b{1,0}, sig{0,1,0,0,0,0,0,0};
  WantedMethods wm;
  wm.m_compiled_bch = wm.m_simple_bch = wm.m_expanded = wm.m_log_of_signature = false; 
  wm.m_compiled_bch = true;
  makeLogSigFunction(3,7,lsf,wm,doNothing);
  /*  lsf.m_f->go(sig.data(),b.data());
  for(double a:sig)
    cout<<a<<" ";
    cout<<endl;*/
}
