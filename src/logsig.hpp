#ifndef LOGSIG_HPP
#define LOGSIG_HPP
#include "bch.hpp"
#include "makeCompiledFunction.hpp"
#include "logSigLength.hpp"
#include<map>

struct LogSigFunction{
  int m_dim, m_level;
  WordPool m_s;
  std::vector<LyndonWord*> m_basisWords;
  FunctionData m_fd;
  std::unique_ptr<FunctionRunner> m_f;
  std::vector<std::vector<float>> m_splitExpandedBasis; //a 2D array, (logsiglevellength * siglevellength), for each level
  std::vector<size_t> m_sigLevelSizes;
};

InputPos inputPosFromSingle(Input i){
  bool isB = i.m_index<0;
  auto a = isB ? InputArr::B : InputArr::A;
  int b = isB ? (-1-i.m_index) : i.m_index-1;
  return std::make_pair(a,b);
}

//makes a vector of uniquified values from in, with indices saying where each of the in elements is represented.
//if "being within tol" isn't transitive on your set of numbers then results will be a bit arbitrary.
void uniquifyDoubles(const std::vector<double>& in, std::vector<size_t>& out_indices, std::vector<double>& out_vals, double tol){
  using P = std::pair<double,std::vector<size_t>>;
  using I = std::vector<P>::iterator;
  std::vector<P> initial;
  initial.reserve(in.size());
  for(size_t i=0; i<in.size(); ++i){
    initial.push_back(P(in[i],{i}));
  }
  std::sort(initial.begin(), initial.end());
  auto end = amalgamate_adjacent(initial.begin(), initial.end(), [tol](P& a, P& b){return std::fabs(a.first-b.first)<tol;},
                                      [](I a, I b){
                                        I a2 = a; 
                                        std::for_each(++a2,b,[a](P& i){a->second.push_back(i.second[0]);});
                                        return true;
                                      });
  //out_vals.reserve(in.size());
  out_vals.clear();
  out_indices.resize(in.size());
  std::for_each(initial.begin(),end,[&](P& p){
      for(size_t i : p.second)
        out_indices[i]=out_vals.size();
      out_vals.push_back(p.first);
    });
}

void makeFunctionDataForBCH(int dim, int level, WordPool& s, FunctionData& fd, std::vector<LyndonWord*>& basisWords,
                            bool justWords, Interrupt interrupt){
  using std::vector;
  using std::make_pair;
  if(level>20)
    throw std::runtime_error("Coefficients only available up to level 20");

  auto wordList = makeListOfLyndonWords(s,dim,level);
  std::unique_ptr<Polynomial> lhs(new Polynomial);
  std::unique_ptr<Polynomial> rhs(new Polynomial);
  if(!justWords)
    for(int i=0; i<dim; ++i)
      rhs->m_data.push_back(std::make_pair(wordList[i],basicCoeff(-i-1))); //just the letters in order
  std::sort(wordList.begin(),wordList.end(),[&s](LyndonWord* a, LyndonWord* b){
      return s.lexicographicLess(a,b);});
  std::stable_sort(wordList.begin(),wordList.end(),[](LyndonWord* a, LyndonWord* b){return a->length()<b->length();});

  if(!justWords)
    for(int i=0; i<(int)wordList.size(); ++i){
      lhs->m_data.push_back(std::make_pair(wordList[i],basicCoeff( i+1)));
    }
  wordList.swap(basisWords);
  if(justWords)
    return;
  std::sort(lhs->m_data.begin(),lhs->m_data.end(),TermLess(s));
  //std::cout<<"bchbefore"<<std::endl;
  //For bch, poly must be lexicographic. For our purposes in this function, we want lexicographic within levels
  auto poly = bch(s,std::move(lhs),std::move(rhs),level, interrupt);
  //std::cout<<"bchdone"<<std::endl;

  fd.m_length_of_b = dim;
  
  using InProduct = vector<Input>;

  vector<vector<const vector<Input>*>> m_neededProducts;
  vector<vector<int>> m_neededProduct_indices;
  m_neededProducts.resize(level);
  m_neededProduct_indices.resize(level);
  //printPolynomial(poly,std::cout);
  //m_neededProducts[i-1] is all the products of i terms which we need.
  for(auto& i : poly.m_data)
    for(auto & v : i.second.m_details){
      m_neededProducts.at(v.first.size()-1).push_back(&v.first);
    }
  for(auto& v : m_neededProducts){
    std::sort(v.begin(), v.end(), [](const InProduct* a, const InProduct* b){return *a<*b;});
    v.erase(std::unique(v.begin(),v.end(),[](const InProduct* a, const InProduct* b){return *a==*b;}),v.end());
  }
  
  InProduct temp;
  for(size_t i = 1; i<m_neededProducts.size(); ++i)
  {
    vector<int>& outputIdx = m_neededProduct_indices[i];
    const auto& prev = m_neededProducts[i-1];
    for(auto& j : m_neededProducts[i]){
      if(i==1){//len==2, so easy
        fd.m_formingT.push_back(std::make_pair(inputPosFromSingle((*j)[0]),inputPosFromSingle((*j)[1])));
      }
      else
      {
        bool found=false;
        for(size_t k=0; k<j->size(); ++k){
          temp = *j;
          temp.erase(temp.begin()+k);
          auto it = std::lower_bound(prev.begin(),prev.end(),temp,[](const InProduct* a, const InProduct& b){return *a<b;});
          if(it!=prev.end() && **it == temp){
            found=true;
            fd.m_formingT.push_back(std::make_pair(inputPosFromSingle((*j)[k]),
                                            std::make_pair(InputArr::T,m_neededProduct_indices[i-1][it-prev.begin()])));
            break;
          }
        }
        if(!found){
         //more exhaustive search not implemented
           std::cout<<"help, not found, level="<<i+1<<std::endl; 
           throw std::runtime_error("I couldn't find a product to make");
        }
      }
      outputIdx.push_back((int)fd.m_formingT.size()-1);
    }
  }

  interrupt();
  std::stable_sort(poly.m_data.begin(), poly.m_data.end(), 
                   [](const Term& a, const Term& b){return a.first->length()<b.first->length();});
  vector<double> wantedConstants;//sorted
  size_t lhs_index=0;
  for(Term& t : poly.m_data){
    //first do the whole thing assuming no uniquification of constants
    Coefficient& c = t.second;
    for(auto& i : c.m_details){
      FunctionData::LineData l;
      l.m_const_offset = (int)wantedConstants.size();
      double constant = i.second;
      if(constant == 0)
        continue;
      wantedConstants.push_back(std::fabs(constant));
      l.m_negative = constant<0;
      l.m_lhs_offset = (int)lhs_index;
      size_t length = i.first.size();
      if(length<2)
        continue;
      const auto& v = m_neededProducts[length-1];
      auto it = std::lower_bound(v.begin(),v.end(),i.first,[](const InProduct* a, const InProduct& b){return *a<b;});
      l.m_rhs_offset = m_neededProduct_indices[length-1][it-v.begin()];
      fd.m_lines.push_back(l);
    }
    ++lhs_index;
  }

  vector<size_t> mapping;
  uniquifyDoubles(wantedConstants,mapping,fd.m_constants,0.00000001);
  for(auto& l : fd.m_lines){
    l.m_const_offset = (int) mapping[l.m_const_offset];
  }
}

class LessLW{
public:
  LessLW(LogSigFunction& lsf):m_lsf(&lsf){}
  bool operator()(const LyndonWord* a, const LyndonWord* b) const{
    return m_lsf->m_s.lexicographicLess(a,b);
  }
  LogSigFunction* m_lsf;
};

void makeSparseLogSigMatrices(int dim, int level, LogSigFunction& lsf, Interrupt interrupt){
  using P = std::pair<size_t,float>;
  //it would be nice to use a lambda and decltype here, but visual studio
  //doesn't allow swapping two lambdas of the same type, 
  //so the vector operations fail.
  std::vector<std::map<const LyndonWord*,std::vector<P>,LessLW> > m;
  m.reserve(level);
  for(int i=0; i<level; ++i)
    m.emplace_back(LessLW(lsf));
  lsf.m_sigLevelSizes.assign(1,(size_t)dim);
  for(int m=2; m<=level; ++m)
    lsf.m_sigLevelSizes.push_back(dim*lsf.m_sigLevelSizes.back());
  for(LyndonWord* w : lsf.m_basisWords){
    if(w->isLetter()){
      m[0][w]={std::make_pair(w->getLetter(),1.0f)};
    }else{
      auto len1 = w->getLeft()->length();
      auto len2 = w->getRight()->length();
      auto& left = m[len1-1][w->getLeft()];
      auto& right = m[len2-1][w->getRight()];
      std::vector<P> v;
      for (const auto& l : left){
        for (const auto& r: right){
          v.push_back(std::make_pair(lsf.m_sigLevelSizes[len2-1]*l.first+r.first,l.second*r.second));
          v.push_back(std::make_pair(lsf.m_sigLevelSizes[len1-1]*r.first+l.first,-l.second*r.second));
        }
      }
      std::sort(v.begin(),v.end());
      v.erase(amalgamate_adjacent_pairs(v.begin(),v.end(),
                                        [](const P& a, const P& b){return a.first==b.first;},
                                        [](P& a, P& b){a.second+=b.second; return a.second!=0;}),
              v.end());
      m[len1+len2-1][w]=std::move(v);
    }
  }
  
  lsf.m_splitExpandedBasis.resize(level);
  for(int lev = 1; lev<=level; ++lev){
    int withinSplitBasisEltOffset=0;
    lsf.m_splitExpandedBasis[lev-1].assign(m[lev-1].size() * lsf.m_sigLevelSizes[lev-1],0);
    for(const auto& p : m[lev-1]){
      for(const auto& q : p.second){
        //pinv this in advance?
        lsf.m_splitExpandedBasis[lev-1][lsf.m_sigLevelSizes[lev-1]*withinSplitBasisEltOffset+q.first]=q.second;
      }
      ++withinSplitBasisEltOffset;
    }
  }
    /* code to print matrices as binary zero/one - looks good in terminal
    for(int lev=1; lev<=level; ++lev){
      for(size_t i=0, j=0; i<m[lev-1].size(); ++i){
        for(size_t k=0; k<lsf.m_sigLevelSizes[lev-1];++k, ++j)
          std::cout<<(std::fabs(lsf.m_splitExpandedBasis[lev-1][j])==0 ? 0 : 1);
        std::cout<<std::endl;
      }
      std::cout<<std::endl;
    }
    */
}

struct WantedMethods{
  bool m_compiled_bch = true;
  bool m_simple_bch = true;
  bool m_log_of_signature = true;
  bool m_expanded = false;
};

//This function sets up the members of lsf. It occasionally calls interrupt
void makeLogSigFunction(int dim, int level, LogSigFunction& lsf, const WantedMethods& wm, Interrupt interrupt){
  using std::vector;
  lsf.m_dim = dim;
  lsf.m_level = level;
  const bool needBCH = wm.m_compiled_bch || wm.m_simple_bch;
  makeFunctionDataForBCH(dim,level,lsf.m_s,lsf.m_fd, lsf.m_basisWords,!needBCH, interrupt);
  interrupt();
  if(wm.m_compiled_bch)
    lsf.m_f.reset(new FunctionRunner(lsf.m_fd));
  else
    lsf.m_f.reset(nullptr);
  interrupt();
  if(wm.m_log_of_signature || wm.m_expanded)
    makeSparseLogSigMatrices(dim,level,lsf,interrupt);
}

//interpret a string as a list of wanted methods, return true on error
bool setWantedMethods(WantedMethods& w, int dim, int level, bool consumer, const std::string& input){
  const auto npos = std::string::npos;
  bool noInput = npos == input.find_first_not_of(" ");
  bool doDefault= (noInput && !consumer) || npos!=input.find_first_of("dD");
  bool defaultIsCompiled = (dim==2 && level<10) || (dim>2 && dim<10 && level < 5);
  bool doEverything = noInput && consumer;
  bool forceCompiled = (defaultIsCompiled && doDefault) || doEverything;
  bool forceLog = (!defaultIsCompiled && doDefault) || doEverything;

  w.m_compiled_bch = forceCompiled || npos!=input.find_first_of("cC");
  w.m_simple_bch=doEverything || (npos!=input.find_first_of("oO"));
  w.m_log_of_signature = forceLog || (npos!=input.find_first_of("sS"));
  w.m_expanded=(npos!=input.find_first_of("xX"));
  return npos!=input.find_first_not_of("cCdDoOsSxX ");
}

const char* const methodError = "Invalid method string. Should be 'd' (default), 'c' (compiled), 'o' (simple BCH object, not compiled), 's' (by taking the log of the signature), or 'x' (to report the expanded log signature), or some combination - order ignored, or None.";

//rename LogSigFunction to LogSigData

#endif
