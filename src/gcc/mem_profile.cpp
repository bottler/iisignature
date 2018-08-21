#include "bch.hpp"
#include "logsig.hpp"
#include "calcSignature.hpp"
#include<fstream>
#include<cstdlib>
#include<cstring>

//usage:

/*
mem_profile 3 4 C L
-calculates the log signature using the C method
-and prints the allocated memory in bytes for the compiled object 
 d=3 and m=4

mem_profile 3 4 X L
- calculates the log signature using the X method and doesn't print anything

both of these are intended to be used inside Massif

L means Lyndon basis, and H would mean Hall basis
*/

using std::vector;
using std::cout;
using std::endl;

using CalcSignature::Signature;
//like in pythonsigs.cpp
void calcSignature(Signature& s2, const double* data, int lengthOfPath, int d, int level){
  Signature s1;

  if(lengthOfPath==1){
    s2.sigOfNothing(d,level);
  }

  vector<double> displacement(d);
  for(int i=1; i<lengthOfPath; ++i){
    for(int j=0;j<d; ++j)
      displacement[j]=data[i*d+j]-data[(i-1)*d+j];
    s1.sigOfSegment(d,level,&displacement[0]);
    if(i==1)
      s2.swap(s1);
    else {
      s2.concatenateWith(d, level, s1);
    }
  }
}

void doNothing(){}

void loadData(){
  const char* file = "../../iisignature_data/bchLyndon20.dat";
  std::ifstream stream (file);
  auto s = new std::string(std::istreambuf_iterator<char>(stream), {});
  g_bchLyndon20_dat = s->c_str();
}  

void doCompiled(int d, int m, LieBasis basis, int pathlength=10){
  loadData();
  if(d==0){
    //special case to find the constant bch overhead
    ReadBCH::read();
    return;
  }
  WantedMethods wantedmethods;
  LogSigFunction lsf(basis);
  WantedMethods wm;
  wm.m_compiled_bch = wm.m_simple_bch = wm.m_expanded = wm.m_log_of_signature = false; 
  wm.m_compiled_bch = true;
  makeLogSigFunction(d,m,lsf,wm,doNothing);
  std::cout << lsf.m_f->m_m.capacity() << "\n";
  //int siglength = calcSigTotalLength(d,m);
  int logsiglength = (int)LogSigLength::countNecklacesUptoLengthM(d,m);
  vector<double> data (d*pathlength), logsig(logsiglength);
  vector<double> b(d);
  for(int i=0; i<pathlength-1; ++i){
    for(int j=0; j<d; ++j)
      b[j]=data[(i+1)*d+j]-data[i*d+j];
    lsf.m_f->go(logsig.data(),b.data());
  }
}
void doProjected(int d, int m, LieBasis basis, int pathlength=10){
  int logsiglength = (int)LogSigLength::countNecklacesUptoLengthM(d,m);
  WantedMethods wantedmethods;
  LogSigFunction lsf(basis);
  WantedMethods wm;
  wm.m_compiled_bch = wm.m_simple_bch = wm.m_expanded = wm.m_log_of_signature = false; 
  wm.m_log_of_signature = true;
  makeLogSigFunction(d,m,lsf,wm,doNothing);
  vector<double> data (d*pathlength), logsig(logsiglength);
  Signature sig;
  calcSignature(sig, &data[0], pathlength, d, m);
  logTensorHorner(sig);
  projectExpandedLogSigToBasis(&logsig[0],&lsf,sig);
}

void usage(const char* name){
  std::cout<<"usage:"<<name<<" D M method basis\n";
  std::cout<<" where D>1 and M>0 and method in {S, C} and basis in {H, L}\n";
  std::cout<<"or 0 0 C H for just memory of loading bch\n";
  std::exit(1);
}

/*
//This can help to predict valgrinds default output file name
void writePidToFile(){
  std::ofstream ofs("mem_profile_pid.txt");
  if(!ofs.is_open()){
    std::cout<<"dying on unable to write pid\n";
    std::exit(1);
  }
  ofs<<getpid();//needs unistd
}
*/

int main(int argc, char** argv){
  bool lyndon = argc>4 && argv[4] && !strcmp("L",argv[4]);
  bool hall = argc>4 && argv[4] && !strcmp("H",argv[4]);
  bool compiled = argc>4 && argv[3] && !strcmp("C",argv[3]);
  bool project = argc>4 && argv[3] && !strcmp("S",argv[3]);
  if(lyndon==hall || compiled==project)
    usage(argv[0]);
  int d = std::atoi(argv[1]);
  int m = std::atoi(argv[2]);
  if((d<2 || m<1) && !(compiled && d==0 && m==0))
    usage(argv[0]);
  LieBasis basis = lyndon ? LieBasis::Lyndon : LieBasis::StandardHall;

  if(compiled)
    doCompiled(d,m,basis);
  if(project)
    doProjected(d,m,basis);
  
}
