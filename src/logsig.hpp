#ifndef LOGSIG_HPP
#define LOGSIG_HPP
#include "bch.hpp"
#include "calcSignature.hpp"
#include "makeCompiledFunction.hpp"
#include "logSigLength.hpp"
#include<map>

//#define SHAREMAT

struct LogSigFunction{
  LogSigFunction(LieBasis basis) : m_s(basis) {}
  int m_dim, m_level;
  BasisPool m_s;
  std::vector<BasisElt*> m_basisElements;
  FunctionData m_fd;
  std::unique_ptr<FunctionRunner> m_f;

  bool canProjectToBasis() const { return m_level < 2 || !m_simples.empty(); }

  //Everything after here is used for the linear algebra calculation of sig -> logsig.
  std::vector<size_t> m_sigLevelSizes;
  std::vector<size_t> m_logLevelSizes;
  int m_siglength = 0;

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

  //In the Lyndon basis, we have a triangularity property,
  //SmallTriangle is the data for a small lower triangular part (block) of the mapping.
  //m_sources is the indices of the relevant elements of the level of the sig
  //m_dests is the indices of the relevant elements of the level of the logsig
  //m_matrix has shape sources*dests, but m_sources and m_dests are actually the same length.
  //m_sources[i] is the index of the word which is the same word as the Lyndon word m_dests[i]
  //If i<j, then m_matrix(i,j) - i.e. m_matrix[i*m_dests.size()+j] - is zero
  //If i==j, then m_matrix(i,j) - i.e. m_matrix[i*m_dests.size()+j] - is one
  struct SmallTriangle{
    std::vector<size_t> m_sources, m_dests;
    std::vector<float> m_matrix;
#ifdef SHAREMAT
    //if m_matrix is empty, look at this element of our element of m_smallTriangles instead
    size_t m_matrixToUse;
#endif
  };
  std::vector<std::vector<SmallTriangle>> m_smallTriangles;
};

InputPos inputPosFromSingle(Input i){
  bool isB = i.m_index<0;
  auto a = isB ? InputArr::B : InputArr::A;
  int b = isB ? (-1-i.m_index) : i.m_index-1;
  return std::make_pair(a,b);
}

//makes a vector of uniquified values from in, with indices saying where each of the in elements is represented.
//I.e. calculates out_vals and out_indices such that out_vals has no repetition and 
//in[i] is approximately the same as out_vals[out_indices[i]].
//If "being within tol" isn't transitive on your set of numbers then results will be a bit arbitrary.
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

void makeFunctionDataForBCH(int dim, int level, BasisPool& s, FunctionData& fd, std::vector<BasisElt*>& basisWords,
                            bool justWords, Interrupt interrupt){
  using std::vector;
  using std::make_pair;

  auto eltList = makeListOfBasisElts(s,dim,level);
  std::unique_ptr<Polynomial> lhs(new Polynomial);
  std::unique_ptr<Polynomial> rhs(new Polynomial);
  if(!justWords){
    if (level>20)
      throw std::runtime_error("Coefficients only available up to level 20");
    rhs->m_data.resize(1);
    for(int i=0; i<dim; ++i)
      rhs->m_data[0].push_back(std::make_pair(eltList[0][i],basicCoeff(-i-1))); //just the letters in order
  }
  for (auto& l : eltList)
    std::sort(l.begin(), l.end());
  if(!justWords){
    lhs->m_data.resize(level);
    for(int l=1,i=1; l<=level; ++l)
      for(auto w : eltList[l-1]){
        lhs->m_data[l-1].push_back(std::make_pair(w,basicCoeff(i++)));
    }
  }
  for(auto& l : eltList)
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

//Given v, a sorted vector<pair<A,B> > which contains (a,x) for exactly one x,
//lookupInFlatMap(v,a) returns x.
template<typename V, typename A>
const typename V::value_type::second_type&
lookupInFlatMap(const V& v, const A& a){
  //if (!std::is_sorted(v.begin(), v.end()))
  //  throw 0;
  //if (!std::is_sorted(v.begin(), v.end(), [](const typename V::value_type& p, 
  //                                      const typename V::value_type& q) {
  //  return p.first < q.first; }))
  //  throw 0;
  auto i = std::lower_bound(v.begin(), v.end(), a,
                            [](const typename V::value_type& p, const A& a){
                              return p.first<a;});
  //if (i == v.end() || i->first != a)
  //  throw 2;
  return i->second;
}

namespace IISignature_algebra {
  using std::vector;
  using std::pair;

  //it would be nice to use a lambda and decltype here, but visual studio
  //doesn't allow swapping two lambdas of the same type, 
  //so the vector operations fail.
  //if x is a basis element for level m, then mappingMatrix[m-1][x] is 
  //the sparse vector which is its image in tensor
  //space - what we call rho(x)
  //a MappingMatrixLevel is a flat map, ordered by address.
  using MappingMatrixLevel =
    vector<pair<const BasisElt*, vector<pair<size_t, float> >>>;
  using MappingMatrix = std::vector<MappingMatrixLevel>;

  void printMappingMatrix(const MappingMatrix& m, std::ostream& of) {
    for (auto& lev : m) {
      for (auto& p : lev) {
        printBasisEltBracketsDigits(*p.first, of);
        of << "\n";
        for (auto& p2 : p.second)
          of << "(" << p2.first << "," << p2.second << ")";
        of << "\n";
      }
      of << "\n";
    }
  }

  MappingMatrix makeMappingMatrix(int /*dim*/, int level, BasisPool &basisPool,
    const std::vector<BasisElt*> &basisWords,
    const std::vector<size_t> &sigLevelSizes) {
    using P = std::pair<size_t, float>;
    using std::vector;
    //it would be nice to use a lambda and decltype here, but visual studio
    //doesn't allow swapping two lambdas of the same type, 
    //so the vector operations fail.
    MappingMatrix m;
    m.resize(level);
    for (BasisElt* w : basisWords) {
      if (w->isLetter()) {
        m[0].emplace_back(w, vector<P>{{w->getLetter(),1.0f} });
      }
      else {
        auto len1 = w->getLeft()->length();
        auto len2 = w->getRight()->length();
        auto& dest = m[len1 + len2 - 1];
        if (dest.empty()) {
          auto& completedLastLevel = m[len1 + len2 - 2];
          std::sort(completedLastLevel.begin(), completedLastLevel.end());
        }
        auto& left = lookupInFlatMap(m[len1 - 1], w->getLeft());
        auto& right = lookupInFlatMap(m[len2 - 1], w->getRight());
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
        dest.emplace_back(w, std::move(v));
      }
    }
    auto& finalLevel = m.back();
    std::sort(finalLevel.begin(), finalLevel.end());
    return m;
  }

  //Given a MappingMatrix (contains the information for the mapping BasisElt -> tensor space
  //fill letterOrderToBE (maps a sorted list of letters to the BEs which have it)
  //and basisEltToIndex (maps a BE to its index, ordered by address)
  //based on level lev
  using Letters_BEs = std::pair<vector<Letter>, vector< const BasisElt*>>;
  using LetterOrderToBE = std::vector<Letters_BEs>;
  using BasisEltToIndex = std::vector<std::pair<const BasisElt*, size_t>>;

  void printMappingMatrixLevelAnalysis(int lev, const LetterOrderToBE& letterOrderToBE, 
    const BasisEltToIndex& basisEltToIndex, std::ostream& of) {
    of << "BEGIN" << lev << "\n";
    for (auto& p : letterOrderToBE) {
      for (auto l : p.first)
        printLetterAsDigit(l, of);
      of << "\n";
      for (auto elt : p.second) {
        printBasisEltBracketsDigits(*elt, of);
        of << "\n";
      }
      of << "\n";
    }
    for (auto& p : basisEltToIndex) {
      printBasisEltBracketsDigits(*p.first, of);
      of << "\n" << p.second <<"\n";
    }
    of << "END" << lev << "\n";
  }

  void analyseMappingMatrixLevel(const MappingMatrix& m, int lev,
    LetterOrderToBE& letterOrderToBE, BasisEltToIndex& basisEltToIndex) {
    letterOrderToBE.reserve(m.size());
    size_t index = 0;
    for (auto& a : m[lev - 1]) {
      const BasisElt* p = a.first;
      std::vector<Letter> l;
      l.reserve(lev);
      p->iterateOverLetters([&l](Letter x) {l.push_back(x); });
      std::sort(l.begin(), l.end());
      letterOrderToBE.push_back(Letters_BEs{ std::move(l),{ p } });
      basisEltToIndex.push_back(std::make_pair(p, index++));
    }
    std::stable_sort(letterOrderToBE.begin(), letterOrderToBE.end(),
      [](const Letters_BEs& a, const Letters_BEs& b) {
      return a.first < b.first;
    });
    std::sort(basisEltToIndex.begin(), basisEltToIndex.end());
    auto end = amalgamate_adjacent(letterOrderToBE.begin(), letterOrderToBE.end(),
      [](Letters_BEs& a, Letters_BEs& b) {return a.first == b.first; },
      [](vector<Letters_BEs>::iterator a, vector<Letters_BEs>::iterator b) {
      auto aa = a;
      ++aa;
      std::for_each(aa, b, [a](Letters_BEs& pp) {
        a->second.push_back(pp.second[0]); });
      return true; });
    letterOrderToBE.erase(end, letterOrderToBE.end());
  }

  std::vector<size_t> getLetterFrequencies(const BasisElt* lw) {
    std::vector<Letter> letters;
    lw->iterateOverLetters([&](Letter l) {letters.push_back(l); });
    std::sort(letters.begin(), letters.end());
    Letter lastLetter = letters[0];
    size_t n = 0;
    std::vector<size_t> out;
    for (Letter l : letters) {
      if (l == lastLetter)
        ++n;
      else {
        out.push_back(n);
        lastLetter = l;
        n = 1;
      }
    }
    out.push_back(n);
    return out;
  }

  class SharedMatrixDetector {
    //list of alphabetical letter frequencies -> pos in m_smallSVDs[lev-1]
    std::map<vector<size_t>, size_t> m_freq2Idx; 
  public:
    //If a calculation has already been done for a basis element with the same letters as w, 
    //put its index in matrixToUse and return false.
    //If not, remember the potentialNewIndex and return true.
    bool need(const BasisElt* w, size_t& matrixToUse, size_t potentialNewIndex) {
      vector<size_t> letterFreqs = getLetterFrequencies(w);
      auto it_bool = m_freq2Idx.insert(std::make_pair(std::move(letterFreqs), potentialNewIndex));
      if (!it_bool.second) {
        matrixToUse = it_bool.first->second;
        return false;
      }
      return true;
    }
  };

  void makeSparseLogSigMatrices(int dim, int level, LogSigFunction& lsf, Interrupt interrupt) {
    using P = std::pair<size_t, float>;
    using std::vector;
    lsf.m_sigLevelSizes.assign(1, (size_t)dim);
    lsf.m_logLevelSizes.assign(1, (size_t)dim);
    lsf.m_siglength = dim;
    for (int m = 2; m <= level; ++m) {
      int a = dim*(int)lsf.m_sigLevelSizes.back();
      lsf.m_siglength += a;
      lsf.m_sigLevelSizes.push_back(a);
    }
    auto m = makeMappingMatrix(dim, level, lsf.m_s, lsf.m_basisElements, lsf.m_sigLevelSizes);
    for (int lev = 2; lev <= level; ++lev)
      lsf.m_logLevelSizes.push_back(m[lev - 1].size());
    //We next write output based on m.
    //The idea is that the coefficient of the basis element x in the log signature
    //is affected by the coefficient of the word y in the signature only if 
    //y and the foliage of x are anagrams. So within each level, we group the basis elements
    //by the alphabetical list of their letters.
    lsf.m_simples.resize(level);
    const bool doTriangles = lsf.m_s.m_basis == LieBasis::Lyndon;
    lsf.m_smallTriangles.resize(level);
    lsf.m_smallSVDs.resize(level);
    for (int lev = 2; lev <= level; ++lev) {
      interrupt();
      LetterOrderToBE letterOrderToBE;
      BasisEltToIndex basisEltToIndex;
      analyseMappingMatrixLevel(m, lev, letterOrderToBE, basisEltToIndex);
      //Now, within lev, letterOrderToBE and basisEltToIndex have been created
#ifdef SHAREMAT
      SharedMatrixDetector detector;
#endif
      for (auto& i : letterOrderToBE) {
        if (1 == i.second.size()) {
          LogSigFunction::Simple sim;
          const BasisElt* lw = i.second[0];
          sim.m_dest = lookupInFlatMap(basisEltToIndex, lw);
          //we can take any we want, so just take the zeroth
          const P& p = lookupInFlatMap(m[lev - 1],lw)[0];
          sim.m_factor = 1.0f / p.second;
          sim.m_source = p.first;
          lsf.m_simples[lev - 1].push_back(sim);
        }
        else if (doTriangles) {
          interrupt();
          lsf.m_smallTriangles[lev - 1].push_back(LogSigFunction::SmallTriangle{});
          LogSigFunction::SmallTriangle& mtx = lsf.m_smallTriangles[lev - 1].back();
          size_t triangleSize = i.second.size();
          vector<intptr_t> fullIdxToSourceIdx(lsf.m_sigLevelSizes[lev - 1], -1);
          for (auto& j : i.second) {
            mtx.m_dests.push_back(lookupInFlatMap(basisEltToIndex, j));
            //The 0 in the next line is relying on the fact that a Lyndon word
            //comes first in its own expansion - Reutenauer theorem 5.1.
            //It also comes with coefficient 1, so the diagonal of mtx.m_matrix
            //will always be 1.
            size_t fullIdx = lookupInFlatMap(m[lev - 1], j).at(0).first;
            mtx.m_sources.push_back(fullIdx);
            fullIdxToSourceIdx.at(fullIdx) = mtx.m_sources.size()-1u;
          }
#ifdef SHAREMAT
          if (detector.need(i.second[0], mtx.m_matrixToUse, lsf.m_smallTriangles[lev - 1].size() - 1))
#endif
          {
            mtx.m_matrix.assign(triangleSize*triangleSize, 0);
            //note that I can use mtx.m_sources to look up the index of each Lyndon word
            for (size_t col = 0; col < triangleSize; ++col) {
              auto lw = i.second[col];
              for (auto& p : lookupInFlatMap(m[lev - 1],lw)){
                intptr_t row = fullIdxToSourceIdx.at(p.first);
                if (row != -1) {
                  mtx.m_matrix[((size_t)row)*triangleSize + col] = p.second;
                }
              }
            }
          }
        }
        else {
          interrupt();
          lsf.m_smallSVDs[lev - 1].push_back(LogSigFunction::SmallSVD{});
          LogSigFunction::SmallSVD& mtx = lsf.m_smallSVDs[lev - 1].back();
          //sourceMap maps expanded (i.e. pos in level of sig) to contracted (i.e. pos in our block)
          //map seems to perform better here than a vector<std::pair<size_t,size_t>>
          std::map<size_t, size_t> sourceMap;
          for (auto& j : i.second) {
            mtx.m_dests.push_back(lookupInFlatMap(basisEltToIndex, j));
            for (auto& k : lookupInFlatMap(m[lev - 1],j)) {
              sourceMap[k.first] = 0;//this element will very often be already present
            }
          }
          size_t idx = 0;
          for (auto& j : sourceMap) {
            j.second = idx++;
            mtx.m_sources.push_back(j.first);
          }
#ifdef SHAREMAT
          if (detector.need(i.second[0], mtx.m_matrixToUse, lsf.m_smallSVDs[lev - 1].size() - 1))
#endif
          {
            mtx.m_matrix.assign(mtx.m_dests.size()*mtx.m_sources.size(), 0);
            for (size_t j = 0; j < i.second.size(); ++j) {
              for (const P& k : lookupInFlatMap(m[lev - 1],i.second[j])) {
                size_t rowBegin = sourceMap[k.first] * mtx.m_dests.size();
                mtx.m_matrix[rowBegin + j] = k.second;
              }
            }
          }
        }
      }

      /*
      for (auto& i : letterOrderToBE) {
        std::cout << i.second.size() << ",";
      }
      std::cout << std::endl;
      for (auto& i : letterOrderToBE) {
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

void projectExpandedLogSigToBasis(double* out, const LogSigFunction* lsf, 
    const CalcSignature::Signature& sig) {
  size_t writeOffset = 0;
  std::vector<CalcSignature::Number> rhs;
  for (auto f : sig.m_data[0])
    out[writeOffset++] = f;
  for (int l = 2; l <= lsf->m_level; ++l) {
    const size_t loglevelLength = lsf->m_logLevelSizes[l - 1];
    for (const auto& i : lsf->m_simples[l - 1]) {
      out[writeOffset + i.m_dest] = sig.m_data[l - 1][i.m_source] * i.m_factor;
    }
    for (auto& i : lsf->m_smallTriangles[l - 1]) {
      size_t triangleSize = i.m_dests.size();
      auto mat =
#ifdef SHAREMAT
        i.m_matrix.empty() ?
        lsf->m_smallTriangles[l - 1][i.m_matrixToUse].m_matrix.data() :
#endif
        i.m_matrix.data();
      for (size_t dest = 0; dest < triangleSize; ++dest) {
        double sum = 0;
        for (size_t source = 0; source < dest; ++source) {
          sum += mat[dest*triangleSize + source] * out[writeOffset + i.m_dests[source]];
        }
        double newVal = sig.m_data[l - 1][i.m_sources[dest]] - sum;
        out[writeOffset + i.m_dests[dest]] = newVal;
      }
    }
    for (auto& i : lsf->m_smallSVDs[l - 1]) {
      auto mat =
#ifdef SHAREMAT
        i.m_matrix.empty() ?
        lsf->m_smallSVDs[l - 1][i.m_matrixToUse].m_matrix.data() :
#endif
        i.m_matrix.data();
      size_t nDests = i.m_dests.size();
      size_t nSources = i.m_sources.size();
      rhs.resize(nSources);
      for (size_t j = 0; j < nSources; ++j)
        rhs[j] = sig.m_data[l - 1][i.m_sources[j]];
      for (size_t j = 0; j < nDests; ++j) {
        double val = 0;
        for (size_t k = 0; k < nSources; ++k)
          val += mat[nSources*j + k] * rhs[k];
        out[writeOffset + i.m_dests[j]] = val;
      }
    }
    writeOffset += loglevelLength;
  }
}

//This backpropagates the function projectExpandedLogSigToBasis.
//Note that, when triangles are in play, projectExpandedLogSigToBasis assumes that its input is 
//a Lie element, which simplifies the calculation. 
//Those assumptions are in play here, so you cannot assume that the output of this function is in
//the subspace of Lie elements. It's not really a "gradient" of a varying expanded log signature.
//This is not a problem for us where all we want to do is backpropagate further 
// - to the signature and then to the path.
void projectExpandedLogSigToBasisBackwards(const double* derivs, const LogSigFunction* lsf,
  CalcSignature::Signature& out) {
  out.sigOfNothing(lsf->m_dim, lsf->m_level);
  size_t writeOffset = 0;
  std::vector<double> rhs;
  for (auto& f : out.m_data[0])
    f = (CalcSignature::Number) derivs[writeOffset++];
  for (int l = 2; l <= lsf->m_level; ++l) {
    const size_t loglevelLength = lsf->m_logLevelSizes[l - 1];
    for (const auto& i : lsf->m_simples[l - 1]) {
      out.m_data[l - 1][i.m_source] = (CalcSignature::Number)(derivs[writeOffset + i.m_dest] * i.m_factor);
    }
    for (auto& i : lsf->m_smallTriangles[l - 1]) {
      size_t triangleSize = i.m_dests.size();
      auto mat =
#ifdef SHAREMAT
        i.m_matrix.empty() ?
        lsf->m_smallTriangles[l - 1][i.m_matrixToUse].m_matrix.data() :
#endif
        i.m_matrix.data();
      rhs.assign(triangleSize, 0.0);
      for (size_t dest = 0; dest < triangleSize; ++dest) {
        for (size_t source = 0; source < dest; ++source) {
          rhs[source] += mat[dest*triangleSize + source] * derivs[writeOffset + i.m_dests[dest]];
        }
      }
      for (size_t source = 0; source < triangleSize; ++source) {
        out.m_data[l - 1][i.m_sources[source]] = (CalcSignature::Number)(derivs[writeOffset + i.m_dests[source]] -rhs[source]);
      }
    }
    for (auto& i : lsf->m_smallSVDs[l - 1]) {
      auto mat =
#ifdef SHAREMAT
        i.m_matrix.empty() ?
        lsf->m_smallSVDs[l - 1][i.m_matrixToUse].m_matrix.data() :
#endif
        i.m_matrix.data();
      size_t nDests = i.m_dests.size();
      size_t nSources = i.m_sources.size();
      rhs.assign(nSources,0.0);
      for (size_t j = 0; j < nDests; ++j) {
        double val = derivs[writeOffset + i.m_dests[j]];
        for (size_t k = 0; k < nSources; ++k)
          rhs[k] += mat[nSources*j + k] * val;
      }
      for (size_t j = 0; j < nSources; ++j)
        out.m_data[l - 1][i.m_sources[j]] = (CalcSignature::Number) rhs[j];
    }
    writeOffset += loglevelLength;
  }
}

struct WantedMethods{
  bool m_compiled_bch = true;
  bool m_simple_bch = true;
  bool m_log_of_signature = true;
  bool m_expanded = false;

  bool m_want_matchCoropa = false;

  const char* m_errMsg = nullptr;
};

//This function sets up the members of lsf. It occasionally calls interrupt
void makeLogSigFunction(int dim, int level, LogSigFunction& lsf, const WantedMethods& wm, Interrupt interrupt){
  using std::vector;
  lsf.m_dim = dim;
  lsf.m_level = level;
  const bool needBCH = wm.m_compiled_bch || wm.m_simple_bch;
  makeFunctionDataForBCH(dim,level,lsf.m_s,lsf.m_fd, lsf.m_basisElements,!needBCH, interrupt);
  interrupt();
  if(wm.m_compiled_bch)
    lsf.m_f.reset(new FunctionRunner(lsf.m_fd));
  else
    lsf.m_f.reset(nullptr);
  interrupt();
  if(wm.m_log_of_signature)
    IISignature_algebra::makeSparseLogSigMatrices(dim,level,lsf,interrupt);
}

const char* const methodError = "Invalid method string. Should be 'D' (default), 'C' (compiled), "
                                "'O' (simple BCH object, not compiled), "
                                "'S' (by taking the log of the signature), "
                                "or 'X' (to report the expanded log signature), "
                                "or some combination - order ignored, possibly with 'H', or None.";

const char* const inconsistentRequest = "You asked for both 'X' and another method in your request, "
                                        "which means I can't tell what output you want.";

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
  bool totallyInvalid = npos!=input.find_first_not_of("cCdDhHoOsSxX ");
  if (totallyInvalid) {
    w.m_errMsg = methodError;
    return true;
  }
  if (consumer && w.m_expanded && (w.m_compiled_bch || w.m_simple_bch || w.m_log_of_signature)) {
    w.m_errMsg = inconsistentRequest;
    return true;
  }
  return false;
}


//rename LogSigFunction to LogSigData

#endif
