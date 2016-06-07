#ifndef LOGSIG_HPP
#define LOGSIG_HPP
#include "bch.h"
#include "makeCompiledFunction.h"

struct LogSigFunction{
  int m_dim, m_level;
  WordPool m_s;
  std::vector<LyndonWord*> m_basisWords;
  FunctionData m_fd;
  std::unique_ptr<FunctionRunner> m_f;
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

void makeFunctionDataForBCH(int dim, int level, WordPool& s, FunctionData& fd, std::vector<LyndonWord*>& basisWords){
  using std::vector;
  using std::make_pair;
  if(level>20)
    throw std::runtime_error("Coefficients only available up to level 20");

  auto wordList = makeListOfLyndonWords(s,dim,level);
  std::unique_ptr<Polynomial> lhs(new Polynomial);
  std::unique_ptr<Polynomial> rhs(new Polynomial);
  for(int i=0; i<dim; ++i)
    rhs->m_data.push_back(std::make_pair(wordList[i],basicCoeff(-i-1))); //just the letters in order
  std::sort(wordList.begin(),wordList.end(),[&s](LyndonWord* a, LyndonWord* b){
      return s.lexicographicLess(a,b);});
  std::stable_sort(wordList.begin(),wordList.end(),[](LyndonWord* a, LyndonWord* b){return a->length()<b->length();});
  
  for(int i=0; i<(int)wordList.size(); ++i){
    lhs->m_data.push_back(std::make_pair(wordList[i],basicCoeff( i+1)));
  }
  wordList.swap(basisWords);
  std::sort(lhs->m_data.begin(),lhs->m_data.end(),TermLess(s));
  //std::cout<<"bchbefore"<<std::endl;
  //For bch, poly must be lexicographic. For our purposes in this function, we want lexicographic within levels
  auto poly = bch(s,std::move(lhs),std::move(rhs),level);
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
      outputIdx.push_back(fd.m_formingT.size()-1);
    }
  }

  std::stable_sort(poly.m_data.begin(), poly.m_data.end(), [](const Term& a, const Term& b){return a.first->length()<b.first->length();});
  vector<double> wantedConstants;//sorted
  size_t lhs_index=0;
  for(Term& t : poly.m_data){
    //first do the whole thing assuming no uniquification of constants
    Coefficient& c = t.second;
    for(auto& i : c.m_details){
      FunctionData::LineData l;
      l.m_const_offset = wantedConstants.size();
      double constant = i.second;
      wantedConstants.push_back(std::fabs(constant));
      l.m_negative = constant<0;
      l.m_lhs_offset = lhs_index;
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
    l.m_const_offset = mapping[l.m_const_offset];
  }
}

void makeLogSigFunction(int dim, int level, LogSigFunction& lsf, bool slow_only){
  lsf.m_dim = dim;
  lsf.m_level = level;
  makeFunctionDataForBCH(dim,level,lsf.m_s,lsf.m_fd, lsf.m_basisWords);
  if(slow_only)
    lsf.m_f.reset(nullptr);
  else
    lsf.m_f.reset(new FunctionRunner(lsf.m_fd));
}

#endif
