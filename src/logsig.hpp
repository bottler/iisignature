#ifndef LOGSIG_HPP
#define LOGSIG_HPP
#include "bch.hpp"
#include "makeCompiledFunction.hpp"
#include "logSigLength.hpp"
#include<map>

struct LogSigFunction{
  LogSigFunction(LieBasis basis) : m_s(basis) {}
  int m_dim, m_level;
  WordPool m_s;
  std::vector<LyndonWord*> m_basisWords;
  FunctionData m_fd;
  std::unique_ptr<FunctionRunner> m_f;

  //Everything after here is used for the linear algebra calculation of sig -> logsig.
  std::vector<size_t> m_sigLevelSizes;
  std::vector<size_t> m_logLevelSizes;

  //A Simple describes how a certain element of a level of a logsig is just
  //a constant multiple of a certain element of a level of a sig.
  struct Simple { size_t m_dest, m_source; float m_factor; };
  std::vector<std::vector<Simple>> m_simples;

  //SmallSVD is the data for a small part (block) of the mapping.
  //m_sources is the indices of the relevant elements of the level of the sig
  //m_dests is the indices of the relevant elements of the level of the logsig
  //m_matrix has shape sources*dests when we fill them.
  //In the python addin, they are replaced with their 
  //pseudoinverses at the end of prepare(),
  //so that it ends up with shape dests*sources
  //Todo as a performance optimization in prepare():
  // -some of these matrices are identical and should be pinv'd once, e.g. those corresponding
  //   to the sets of Lyndon words {1223, 1232, 1322} and {1224, 1242, 1422}
  //   - not convincingly quicker, perhaps because it takes work to find these cases
  //     and there are not so many of them. This is commented out with #ifdef SHAREMAT.
  // -beyond that, some are permutation similar, e.g. the above and {1123, 1132, 1213}, and
  //   could also be shared.
  struct SmallSVD { 
    std::vector<size_t> m_sources, m_dests; 
    std::vector<float> m_matrix; 
#ifdef SHAREMAT
    //if m_matrix is empty, look at this element of our element of m_smallSVDs instead
    size_t m_matrixToUse;
#endif
  };
  std::vector<std::vector<SmallSVD>> m_smallSVDs;

};

InputPos inputPosFromSingle(Input i){
  bool isB = i.m_index<0;
  auto a = isB ? InputArr::B : InputArr::A;
  int b = isB ? (-1-i.m_index) : i.m_index-1;
  return std::make_pair(a,b);
}

//makes a vector of uniquified values from in, with indices saying where each of the in elements is represented.
//if "being within tol" isn't transitive on your set of numbers then results will be a bit arbitrary.
void uniquifyDoubles(const std::vector<double>& in, std::vector<size_t>& out_indices,
                    std::vector<double>& out_vals, double tol){
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
  if(!justWords){
    rhs->m_data.resize(1);
    for(int i=0; i<dim; ++i)
      rhs->m_data[0].push_back(std::make_pair(wordList[0][i],basicCoeff(-i-1))); //just the letters in order
  }
  for(auto& l : wordList)
    std::sort(l.begin(),l.end(),[&s](LyndonWord* a, LyndonWord* b){
      return s.lexicographicLess(a,b);});
  if(!justWords){
    lhs->m_data.resize(level);
    for(int l=1,i=1; l<=level; ++l)
      for(auto w : wordList[l-1]){
        lhs->m_data[l-1].push_back(std::make_pair(w,basicCoeff(i++)));
    }
  }
  for(auto& l : wordList)
    for(auto& k : l)
      basisWords.push_back(k);
  
  if(justWords)
    return;
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
  for(auto& l : poly.m_data)
    for(auto& i : l)
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
          auto it = std::lower_bound(prev.begin(),prev.end(),temp,
                                     [](const InProduct* a, const InProduct& b){return *a<b;});
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
  vector<double> wantedConstants;//sorted
  size_t lhs_index=0;
  for(auto& tt : poly.m_data)
    for(Term& t : tt){
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
        auto it = std::lower_bound(v.begin(),v.end(),i.first,
                          [](const InProduct* a, const InProduct& b){return *a<b;});
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
  LessLW(WordPool& wp):m_wp(&wp){}
  bool operator()(const LyndonWord* a, const LyndonWord* b) const{
    return m_wp->lexicographicLess(a,b);
  }
  WordPool* m_wp;
};

//Given v, a sorted vector<pair<A,B> > which contains (a,x) for exactly one x,
//lookupInFlatMap(v,a) returns x.
template<typename V, typename A>
const typename V::value_type::second_type&
lookupInFlatMap(const V& v, const A& a){
  auto i = std::lower_bound(v.begin(), v.end(), a, 
                            [](const typename V::value_type& p, const A& a){
                              return p.first<a;});
  return i->second;
}

std::vector<size_t> getLetterFrequencies(const LyndonWord* lw){
  std::vector<Letter> letters;
  lw->iterateOverLetters([&](Letter l){letters.push_back(l);});
  std::sort(letters.begin(),letters.end());
  Letter lastLetter=letters[0];
  size_t n = 0;
  std::vector<size_t> out;
  for(Letter l : letters){
    if(l==lastLetter)
      ++n;
    else{
      out.push_back(n);
      lastLetter=l;
      n=1;
    }
  }
  out.push_back(n);
  return out;
}

namespace IISignature_algebra {
  using std::vector;
  using std::pair;

  //it would be nice to use a lambda and decltype here, but visual studio
  //doesn't allow swapping two lambdas of the same type, 
  //so the vector operations fail.
  //if x is a lyndon word, then mappingMatrix[len(x)-1][x] is the sparse vector which is its image in tensor
  //space - what we call rho(x)
  using MappingMatrixLevel =
    std::map<const LyndonWord*, std::vector<std::pair<size_t, float> >, LessLW>;
  using MappingMatrix = std::vector<MappingMatrixLevel>;

  MappingMatrix makeMappingMatrix(int /*dim*/, int level, WordPool &wp,
    const std::vector<LyndonWord*> &basisWords,
    const std::vector<size_t> &sigLevelSizes) {
    using P = std::pair<size_t, float>;
    using std::vector;
    //it would be nice to use a lambda and decltype here, but visual studio
    //doesn't allow swapping two lambdas of the same type, 
    //so the vector operations fail.
    //if x is a lyndon word, then m[len(x)-1][x] is the sparse vector which is its image in tensor
    //space - what we call rho(x)
    std::vector<std::map<const LyndonWord*, std::vector<P>, LessLW> > m;
    m.reserve(level);
    for (int i = 0; i < level; ++i)
      m.emplace_back(LessLW(wp));
    for (LyndonWord* w : basisWords) {
      if (w->isLetter()) {
        m[0][w] = { std::make_pair(w->getLetter(),1.0f) };
      }
      else {
        auto len1 = w->getLeft()->length();
        auto len2 = w->getRight()->length();
        auto& left = m[len1 - 1][w->getLeft()];
        auto& right = m[len2 - 1][w->getRight()];
        std::vector<P> v;
        for (const auto& l : left) {
          for (const auto& r : right) {
            v.push_back(std::make_pair(sigLevelSizes[len2 - 1] * l.first + r.first, l.second*r.second));
            v.push_back(std::make_pair(sigLevelSizes[len1 - 1] * r.first + l.first, -l.second*r.second));
          }
        }
        std::sort(v.begin(), v.end());
        v.erase(amalgamate_adjacent_pairs(v.begin(), v.end(),
          [](const P& a, const P& b) {return a.first == b.first; },
          [](P& a, P& b) {a.second += b.second; return a.second != 0; }),
          v.end());
        m[len1 + len2 - 1][w] = std::move(v);
      }
    }
    return m;
  }

  //Given a MappingMatrix (contains the information for  the mapping LW -> tensor space
  //fill letterOrderToLW (maps a sorted list of letters to the LWs which have it)
  //and lyndonWordToIndex (maps an LW to its index)
  //based on level lev
  using Letters_LWs = std::pair<vector<Letter>, vector< const LyndonWord*>>;
  using LetterOrderToLW = std::vector<Letters_LWs>;
  using LyndonWordToIndex = std::vector<std::pair<const LyndonWord*, size_t>>;
  void analyseMappingMatrixLevel(const MappingMatrix& m, int lev,
    LetterOrderToLW& letterOrderToLW, LyndonWordToIndex& lyndonWordToIndex) {
    letterOrderToLW.reserve(m.size());
    size_t index = 0;
    for (auto& a : m[lev - 1]) {
      const LyndonWord* p = a.first;
      std::vector<Letter> l;
      l.reserve(lev);
      p->iterateOverLetters([&l](Letter x) {l.push_back(x); });
      std::sort(l.begin(), l.end());
      letterOrderToLW.push_back(Letters_LWs{ std::move(l),{ p } });
      lyndonWordToIndex.push_back(std::make_pair(p, index++));
    }
    std::sort(letterOrderToLW.begin(), letterOrderToLW.end());
    std::sort(lyndonWordToIndex.begin(), lyndonWordToIndex.end());
    auto end = amalgamate_adjacent(letterOrderToLW.begin(), letterOrderToLW.end(),
      [](Letters_LWs& a, Letters_LWs& b) {return a.first == b.first; },
      [](vector<Letters_LWs>::iterator a, vector<Letters_LWs>::iterator b) {
      auto aa = a;
      ++aa;
      std::for_each(aa, b, [a](Letters_LWs& pp) {
        a->second.push_back(pp.second[0]); });
      return true; });
    letterOrderToLW.erase(end, letterOrderToLW.end());
  }

  void makeSparseLogSigMatrices(int dim, int level, LogSigFunction& lsf, Interrupt interrupt) {
    using P = std::pair<size_t, float>;
    using std::vector;
    lsf.m_sigLevelSizes.assign(1, (size_t)dim);
    lsf.m_logLevelSizes.assign(1, (size_t)dim);
    for (int m = 2; m <= level; ++m)
      lsf.m_sigLevelSizes.push_back(dim*lsf.m_sigLevelSizes.back());
    auto m = makeMappingMatrix(dim, level, lsf.m_s, lsf.m_basisWords, lsf.m_sigLevelSizes);
    for (int lev = 2; lev <= level; ++lev)
      lsf.m_logLevelSizes.push_back(m[lev - 1].size());
    //We next write output based on m.
    //The idea is that the coefficient of the Lyndon word x in the log signature
    //is affected by the coefficient of the word y in the signature only if 
    //x and y are anagrams. So within each level, we group the Lyndon words
    //by the alphabetical list of their letters.
    lsf.m_simples.resize(level);
    lsf.m_smallSVDs.resize(level);
    for (int lev = 2; lev <= level; ++lev) {
      interrupt();
      LetterOrderToLW letterOrderToLW;
      LyndonWordToIndex lyndonWordToIndex;
      analyseMappingMatrixLevel(m, lev, letterOrderToLW, lyndonWordToIndex);
      //Now, within lev, letterOrderToLW and lyndonWordToIndex have been created
#ifdef SHAREMAT
      std::map<vector<size_t>, size_t> freq2Idx; //list of alphabetical letter frequencies -> pos in m_smallSVDs[lev-1]
#endif
      for (auto& i : letterOrderToLW) {
        if (1 == i.second.size()) {
          LogSigFunction::Simple sim;
          const LyndonWord* lw = i.second[0];
          sim.m_dest = lookupInFlatMap(lyndonWordToIndex, lw);
          //we can take any we want, so just take the zeroth
          const P& p = m[lev - 1].at(lw)[0];
          sim.m_factor = 1.0f / p.second;
          sim.m_source = p.first;
          lsf.m_simples[lev - 1].push_back(sim);
        }
        else {
          interrupt();
          lsf.m_smallSVDs[lev - 1].push_back(LogSigFunction::SmallSVD{});
          LogSigFunction::SmallSVD& mtx = lsf.m_smallSVDs[lev - 1].back();
          //sourceMap maps expanded (i.e. pos in level of sig) to contracted (i.e. pos in our block)
          //map seems to perform better here than a vector<std::pair<size_t,size_t>>
          std::map<size_t, size_t> sourceMap;
          for (auto& j : i.second) {
            mtx.m_dests.push_back(lookupInFlatMap(lyndonWordToIndex, j));
            for (auto& k : m[lev - 1].at(j)) {
              sourceMap[k.first] = 0;//this element will very often be already present
            }
          }
          size_t idx = 0;
          for (auto& j : sourceMap) {
            j.second = idx++;
            mtx.m_sources.push_back(j.first);
          }
#ifdef SHAREMAT
          vector<size_t> letterFreqs = getLetterFrequencies(i.second[0]);
          size_t dummy = 0;
          auto it_bool = freq2Idx.insert(std::make_pair(std::move(letterFreqs), dummy));
          if (!it_bool.second)
            mtx.m_matrixToUse = it_bool.first->second;
          else {
            it_bool.first->second = lsf.m_smallSVDs[lev - 1].size() - 1;
#endif
            mtx.m_matrix.assign(mtx.m_dests.size()*mtx.m_sources.size(), 0);
            for (size_t j = 0; j < i.second.size(); ++j) {
              for (P& k : m[lev - 1].at(i.second[j])) {
                size_t rowBegin = sourceMap[k.first] * mtx.m_dests.size();
                mtx.m_matrix[rowBegin + j] = k.second;
              }
            }
#ifdef SHAREMAT
          }
#endif
        }
      }

      /*
      for (auto& i : letterOrderToLW) {
        std::cout << i.second.size() << ",";
      }
      std::cout << std::endl;
      for (auto& i : letterOrderToLW) {
        if (i.second.size() > 130) {
          std::cout << i.second.size() << ",";
          for (Letter l : i.first) {
            printLetterAsDigit(l, std::cout);
          }
          std::cout << "\n";
        }
      }
      */
    }
  }
}

struct WantedMethods{
  bool m_compiled_bch = true;
  bool m_simple_bch = true;
  bool m_log_of_signature = true;
  bool m_expanded = false;

  bool m_want_matchCoropa = false;
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
  if(wm.m_log_of_signature)
    IISignature_algebra::makeSparseLogSigMatrices(dim,level,lsf,interrupt);
}

//interpret a string as a list of wanted methods, return true on error
bool setWantedMethods(WantedMethods& w, int dim, int level, bool consumer, const std::string& input){
  const auto npos = std::string::npos;
  bool noInput = npos == input.find_first_of("cCdDoOsSxX"); //no method (DCOSX) is given
  bool doDefault= (noInput && !consumer) || npos!=input.find_first_of("dD");
  bool defaultIsCompiled = (dim==2 && level<10) || (dim>2 && dim<10 && level < 5);
  bool doEverything = noInput && consumer;
  bool forceCompiled = (defaultIsCompiled && doDefault) || doEverything;
  bool forceLog = (!defaultIsCompiled && doDefault) || doEverything;

  w.m_compiled_bch = forceCompiled || npos!=input.find_first_of("cC");
  w.m_simple_bch=doEverything || (npos!=input.find_first_of("oO"));
  w.m_log_of_signature = forceLog || (npos!=input.find_first_of("sS"));
  w.m_expanded = (npos != input.find_first_of("xX"));
  w.m_want_matchCoropa = (npos != input.find_first_of("hH"));
  return npos!=input.find_first_not_of("cCdDhHoOsSxX ");
}

const char* const methodError = "Invalid method string. Should be 'D' (default), 'C' (compiled), "
                                "'O' (simple BCH object, not compiled), "
                                "'S' (by taking the log of the signature), "
                                "or 'X' (to report the expanded log signature), "
                                "or some combination - order ignored, possibly with 'H', or None.";

//rename LogSigFunction to LogSigData

#endif
